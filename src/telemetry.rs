use std::{
    collections::HashSet,
    process::{Command, Stdio},
};

#[cfg(target_os = "linux")]
use std::fs;

use anyhow::{Context, Result};

#[derive(Debug, Clone, PartialEq)]
pub struct ResourceSnapshot {
    pub process_count: usize,
    pub cpu_percent: f64,
    pub rss_kib: u64,
    pub gpu_memory_mib: Option<u64>,
    pub disk_free_mib: Option<u64>,
    pub network_rx_bytes: Option<u64>,
    pub network_tx_bytes: Option<u64>,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct GpuTelemetry {
    pub name: String,
    pub total_vram_mib: u64,
    pub used_vram_mib: u64,
    pub free_vram_mib: u64,
    pub gpu_util_percent: u32,
    pub mem_util_percent: u32,
    pub temperature_c: u32,
    pub power_watts: f64,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct SystemMemoryTelemetry {
    pub total_ram_mib: u64,
    pub available_ram_mib: u64,
    pub used_ram_mib: u64,
    pub total_swap_mib: u64,
    pub free_swap_mib: u64,
    pub used_swap_mib: u64,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct ServerIoTelemetry {
    pub pid: u32,
    pub read_bytes: u64,
    pub read_rate_bytes_per_sec: Option<f64>,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct SystemTelemetrySnapshot {
    pub resource_summary: String,
    pub gpu: Option<GpuTelemetry>,
    pub memory: Option<SystemMemoryTelemetry>,
    pub server_io: Option<ServerIoTelemetry>,
}

impl ResourceSnapshot {
    pub fn render(&self, subject: &str, elapsed_seconds: Option<u64>) -> String {
        let gpu = self
            .gpu_memory_mib
            .map(|memory| format!("{memory} MiB"))
            .unwrap_or_else(|| "n/a".to_owned());
        let elapsed = elapsed_seconds.map_or_else(String::new, |seconds| {
            format!(" elapsed={:02}:{:02}", seconds / 60, seconds % 60)
        });
        let disk = self
            .disk_free_mib
            .map_or_else(|| "n/a".to_owned(), |value| format!("{value} MiB"));
        let network = match (self.network_rx_bytes, self.network_tx_bytes) {
            (Some(rx), Some(tx)) => format!("rx={rx}B tx={tx}B"),
            _ => "n/a".to_owned(),
        };
        format!(
            "Resources: {subject}={} cpu={:.1}% ram={:.1} MiB gpu={gpu} disk={disk} net={network}{elapsed}",
            self.process_count,
            self.cpu_percent,
            self.rss_kib as f64 / 1024.0,
        )
    }
}

#[derive(Debug, Clone, PartialEq)]
struct ProcessSample {
    pid: u32,
    parent_pid: u32,
    process_group: u32,
    cpu_percent: f64,
    rss_kib: u64,
    command: String,
}

pub fn snapshot_process_group(process_group: u32) -> Result<ResourceSnapshot> {
    let processes = read_process_table()?;
    let pids = processes
        .iter()
        .filter(|process| process.process_group == process_group)
        .map(|process| process.pid)
        .collect::<HashSet<_>>();
    Ok(aggregate(&processes, &pids))
}

pub fn snapshot_descendants(parent_pid: u32) -> Result<ResourceSnapshot> {
    let processes = read_process_table()?;
    let pids = descendant_pids(&processes, parent_pid);
    Ok(aggregate(&processes, &pids))
}

pub fn find_process_named(name: &str) -> Result<Option<u32>> {
    let processes = read_process_table()?;
    Ok(processes
        .iter()
        .find(|process| command_name(&process.command) == name)
        .map(|process| process.pid))
}

fn read_process_table() -> Result<Vec<ProcessSample>> {
    let output = Command::new("ps")
        .args(["-axo", "pid=,ppid=,pgid=,pcpu=,rss=,comm="])
        .env("LC_ALL", "C")
        .stdin(Stdio::null())
        .stderr(Stdio::null())
        .output()
        .context("run ps for resource telemetry")?;
    anyhow::ensure!(output.status.success(), "ps exited with {}", output.status);
    Ok(parse_process_table(&String::from_utf8_lossy(
        &output.stdout,
    )))
}

fn parse_process_table(output: &str) -> Vec<ProcessSample> {
    output.lines().filter_map(parse_process_row).collect()
}

fn parse_process_row(row: &str) -> Option<ProcessSample> {
    let mut fields = row.split_whitespace();
    let pid = fields.next()?.parse().ok()?;
    let parent_pid = fields.next()?.parse().ok()?;
    let process_group = fields.next()?.parse().ok()?;
    let cpu_percent = fields.next()?.parse().ok()?;
    let rss_kib = fields.next()?.parse().ok()?;
    let command = fields.collect::<Vec<_>>().join(" ");
    if command.is_empty() {
        return None;
    }
    Some(ProcessSample {
        pid,
        parent_pid,
        process_group,
        cpu_percent,
        rss_kib,
        command,
    })
}

fn descendant_pids(processes: &[ProcessSample], parent_pid: u32) -> HashSet<u32> {
    let mut descendants = HashSet::new();
    loop {
        let before = descendants.len();
        for process in processes {
            if process.parent_pid == parent_pid || descendants.contains(&process.parent_pid) {
                descendants.insert(process.pid);
            }
        }
        if descendants.len() == before {
            return descendants;
        }
    }
}

fn aggregate(processes: &[ProcessSample], pids: &HashSet<u32>) -> ResourceSnapshot {
    let cpu_percent = processes
        .iter()
        .filter(|process| pids.contains(&process.pid))
        .map(|process| process.cpu_percent)
        .sum();
    let rss_kib = processes
        .iter()
        .filter(|process| pids.contains(&process.pid))
        .map(|process| process.rss_kib)
        .sum();
    ResourceSnapshot {
        process_count: pids.len(),
        cpu_percent,
        rss_kib,
        gpu_memory_mib: gpu_memory_mib(pids),
        disk_free_mib: disk_free_mib("."),
        network_rx_bytes: network_bytes().map(|bytes| bytes.0),
        network_tx_bytes: network_bytes().map(|bytes| bytes.1),
    }
}

fn gpu_memory_mib(pids: &HashSet<u32>) -> Option<u64> {
    if pids.is_empty() {
        return None;
    }
    let output = Command::new("nvidia-smi")
        .args([
            "--query-compute-apps=pid,used_memory",
            "--format=csv,noheader,nounits",
        ])
        .env("LC_ALL", "C")
        .stdin(Stdio::null())
        .stderr(Stdio::null())
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    Some(parse_gpu_memory(
        &String::from_utf8_lossy(&output.stdout),
        pids,
    ))
}

fn parse_gpu_memory(output: &str, pids: &HashSet<u32>) -> u64 {
    output
        .lines()
        .filter_map(|row| {
            let (pid, memory) = row.split_once(',')?;
            let pid = pid.trim().parse::<u32>().ok()?;
            let memory = memory.trim().parse::<u64>().ok()?;
            pids.contains(&pid).then_some(memory)
        })
        .sum()
}

fn command_name(command: &str) -> &str {
    command.rsplit('/').next().unwrap_or(command)
}

fn disk_free_mib(path: &str) -> Option<u64> {
    let output = Command::new("df")
        .args(["-k", path])
        .env("LC_ALL", "C")
        .stdin(Stdio::null())
        .stderr(Stdio::null())
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    String::from_utf8_lossy(&output.stdout)
        .lines()
        .skip(1)
        .find_map(|line| line.split_whitespace().nth(3)?.parse::<u64>().ok())
        .map(|kib| kib / 1024)
}

fn network_bytes() -> Option<(u64, u64)> {
    #[cfg(target_os = "linux")]
    {
        let text = fs::read_to_string("/proc/net/dev").ok()?;
        parse_proc_net_dev(&text)
    }
    #[cfg(not(target_os = "linux"))]
    {
        None
    }
}

#[cfg(target_os = "linux")]
fn parse_proc_net_dev(text: &str) -> Option<(u64, u64)> {
    let mut rx = 0_u64;
    let mut tx = 0_u64;
    let mut found = false;
    for line in text.lines().skip(2) {
        let (_, values) = line.split_once(':')?;
        let fields = values.split_whitespace().collect::<Vec<_>>();
        if fields.len() < 9 {
            continue;
        }
        let Some(line_rx) = fields[0].parse::<u64>().ok() else {
            continue;
        };
        let Some(line_tx) = fields[8].parse::<u64>().ok() else {
            continue;
        };
        rx = rx.saturating_add(line_rx);
        tx = tx.saturating_add(line_tx);
        found = true;
    }
    found.then_some((rx, tx))
}

pub fn query_gpu_telemetry() -> Option<GpuTelemetry> {
    let output = Command::new("nvidia-smi")
        .args([
            "--query-gpu=gpu_name,memory.total,memory.used,memory.free,utilization.gpu,utilization.memory,temperature.gpu,power.draw",
            "--format=csv,noheader,nounits",
        ])
        .env("LC_ALL", "C")
        .stdin(Stdio::null())
        .stderr(Stdio::null())
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    parse_nvidia_smi_gpu_query(&String::from_utf8_lossy(&output.stdout))
}

pub fn parse_nvidia_smi_gpu_query(output: &str) -> Option<GpuTelemetry> {
    for line in output.lines() {
        let fields = line.split(',').map(str::trim).collect::<Vec<_>>();
        if fields.len() < 8 {
            continue;
        }
        let name = fields[0].to_string();
        let total_vram_mib = fields[1].parse().unwrap_or(0);
        let used_vram_mib = fields[2].parse().unwrap_or(0);
        let free_vram_mib = fields[3].parse().unwrap_or(0);
        let gpu_util_percent = fields[4].parse().unwrap_or(0);
        let mem_util_percent = fields[5].parse().unwrap_or(0);
        let temperature_c = fields[6].parse().unwrap_or(0);
        let power_watts = fields[7].parse().unwrap_or(0.0);
        if total_vram_mib > 0 || !name.is_empty() {
            return Some(GpuTelemetry {
                name,
                total_vram_mib,
                used_vram_mib,
                free_vram_mib,
                gpu_util_percent,
                mem_util_percent,
                temperature_c,
                power_watts,
            });
        }
    }
    None
}

pub fn query_system_memory() -> Option<SystemMemoryTelemetry> {
    #[cfg(target_os = "linux")]
    {
        let text = fs::read_to_string("/proc/meminfo").ok()?;
        parse_proc_meminfo(&text)
    }
    #[cfg(not(target_os = "linux"))]
    {
        None
    }
}

pub fn parse_proc_meminfo(text: &str) -> Option<SystemMemoryTelemetry> {
    let mut total_ram_kib = 0_u64;
    let mut avail_ram_kib = 0_u64;
    let mut total_swap_kib = 0_u64;
    let mut free_swap_kib = 0_u64;
    for line in text.lines() {
        let Some((key, val_str)) = line.split_once(':') else {
            continue;
        };
        let num = val_str
            .split_whitespace()
            .next()
            .and_then(|s| s.parse::<u64>().ok())
            .unwrap_or(0);
        match key.trim() {
            "MemTotal" => total_ram_kib = num,
            "MemAvailable" => avail_ram_kib = num,
            "SwapTotal" => total_swap_kib = num,
            "SwapFree" => free_swap_kib = num,
            _ => {}
        }
    }
    if total_ram_kib == 0 {
        return None;
    }
    let total_ram_mib = total_ram_kib / 1024;
    let available_ram_mib = avail_ram_kib / 1024;
    let used_ram_mib = total_ram_mib.saturating_sub(available_ram_mib);
    let total_swap_mib = total_swap_kib / 1024;
    let free_swap_mib = free_swap_kib / 1024;
    let used_swap_mib = total_swap_mib.saturating_sub(free_swap_mib);
    Some(SystemMemoryTelemetry {
        total_ram_mib,
        available_ram_mib,
        used_ram_mib,
        total_swap_mib,
        free_swap_mib,
        used_swap_mib,
    })
}

pub fn query_server_io(
    pid: u32,
    prev_read_bytes: Option<u64>,
    elapsed_secs: Option<f64>,
) -> Option<ServerIoTelemetry> {
    #[cfg(target_os = "linux")]
    {
        let path = format!("/proc/{pid}/io");
        let text = fs::read_to_string(&path).ok()?;
        let read_bytes = parse_proc_io(&text)?;
        let read_rate_bytes_per_sec = match (prev_read_bytes, elapsed_secs) {
            (Some(prev), Some(elapsed)) if elapsed > 0.0 => {
                let delta = read_bytes.saturating_sub(prev);
                Some(delta as f64 / elapsed)
            }
            _ => None,
        };
        Some(ServerIoTelemetry {
            pid,
            read_bytes,
            read_rate_bytes_per_sec,
        })
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = (pid, prev_read_bytes, elapsed_secs);
        None
    }
}

pub fn parse_proc_io(text: &str) -> Option<u64> {
    for line in text.lines() {
        if let Some((key, val)) = line.split_once(':') {
            if key.trim() == "read_bytes" {
                return val.trim().parse::<u64>().ok();
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_processes() -> Vec<ProcessSample> {
        parse_process_table(
            " 100 1 100 1.5 1024 /usr/bin/llama-swap\n\
             101 100 100 12.5 4096 /opt/llama-server\n\
             102 101 100 3.0 2048 /opt/worker\n\
             200 1 200 99.0 9999 /bin/other\n\
             malformed row\n",
        )
    }

    #[test]
    fn parses_process_rows_and_skips_invalid_input() {
        let processes = sample_processes();
        assert_eq!(processes.len(), 4);
        assert_eq!(processes[0].pid, 100);
        assert_eq!(processes[1].cpu_percent, 12.5);
        assert_eq!(processes[2].command, "/opt/worker");
    }

    #[test]
    fn discovers_all_descendant_generations() {
        let descendants = descendant_pids(&sample_processes(), 100);
        assert_eq!(descendants, HashSet::from([101, 102]));
    }

    #[test]
    fn aggregates_only_selected_processes() {
        let processes = sample_processes();
        let snapshot = aggregate_without_gpu(&processes, &HashSet::from([100, 101, 102]));
        assert_eq!(snapshot.process_count, 3);
        assert!((snapshot.cpu_percent - 17.0).abs() < f64::EPSILON);
        assert_eq!(snapshot.rss_kib, 7168);
        assert_eq!(
            snapshot.render("procs", Some(65)),
            "Resources: procs=3 cpu=17.0% ram=7.0 MiB gpu=n/a disk=n/a net=n/a elapsed=01:05"
        );
    }

    #[test]
    fn sums_nvidia_rows_for_selected_pids() {
        let pids = HashSet::from([101, 102]);
        assert_eq!(
            parse_gpu_memory("101, 1000\n102, 250\n200, 999\nbad\n", &pids),
            1250
        );
    }

    #[test]
    fn matches_executable_basename() {
        assert_eq!(command_name("/opt/bin/llama-swap"), "llama-swap");
        assert_eq!(command_name("llama-swap"), "llama-swap");
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn parses_proc_network_counters() {
        let text = "Inter-| Receive | Transmit\n face |bytes packets errs drop fifo frame compressed multicast|bytes packets errs drop fifo colls carrier compressed\n eth0: 100 1 0 0 0 0 0 0 200 2 0 0 0 0 0 0\n";
        assert_eq!(parse_proc_net_dev(text), Some((100, 200)));
    }

    #[test]
    fn parses_nvidia_smi_gpu_query() {
        let sample = "NVIDIA GeForce RTX 4070, 12282, 11678, 225, 42, 15, 44, 115.5\n";
        let gpu = parse_nvidia_smi_gpu_query(sample).expect("valid gpu parse");
        assert_eq!(gpu.name, "NVIDIA GeForce RTX 4070");
        assert_eq!(gpu.total_vram_mib, 12282);
        assert_eq!(gpu.used_vram_mib, 11678);
        assert_eq!(gpu.free_vram_mib, 225);
        assert_eq!(gpu.gpu_util_percent, 42);
        assert_eq!(gpu.mem_util_percent, 15);
        assert_eq!(gpu.temperature_c, 44);
        assert!((gpu.power_watts - 115.5).abs() < f64::EPSILON);
    }

    #[test]
    fn parses_proc_meminfo() {
        let text = "MemTotal:       64629304 kB\n\
                    MemFree:          886480 kB\n\
                    MemAvailable:   55785092 kB\n\
                    SwapTotal:      20971516 kB\n\
                    SwapFree:       15728636 kB\n";
        let mem = parse_proc_meminfo(text).expect("valid meminfo parse");
        assert_eq!(mem.total_ram_mib, 63114);
        assert_eq!(mem.available_ram_mib, 54477);
        assert_eq!(mem.used_ram_mib, 8637);
        assert_eq!(mem.total_swap_mib, 20479);
        assert_eq!(mem.free_swap_mib, 15359);
        assert_eq!(mem.used_swap_mib, 5120);
    }

    #[test]
    fn parses_proc_io() {
        let text = "rchar: 122905434\n\
                    wchar: 5192\n\
                    syscr: 30397\n\
                    syscw: 150\n\
                    read_bytes: 25495613440\n\
                    write_bytes: 24576\n";
        assert_eq!(parse_proc_io(text), Some(25495613440));
    }

    fn aggregate_without_gpu(processes: &[ProcessSample], pids: &HashSet<u32>) -> ResourceSnapshot {
        ResourceSnapshot {
            process_count: pids.len(),
            cpu_percent: processes
                .iter()
                .filter(|process| pids.contains(&process.pid))
                .map(|process| process.cpu_percent)
                .sum(),
            rss_kib: processes
                .iter()
                .filter(|process| pids.contains(&process.pid))
                .map(|process| process.rss_kib)
                .sum(),
            gpu_memory_mib: None,
            disk_free_mib: None,
            network_rx_bytes: None,
            network_tx_bytes: None,
        }
    }
}
