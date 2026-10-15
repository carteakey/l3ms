"""Validate API graphs against installed native node source, without importing it."""
import ast
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
COMFY = ROOT / "vendor/ComfyUI"
RECIPES = {
    "yue2": ROOT / "workflows/yue2/audio_yue2_text2music_api.json",
    "ltx": ROOT / "workflows/ltx-2.5/video_ltx2_5_t2v_preview_api.json",
}
SOURCES = [
    "nodes.py", "comfy_extras/nodes_yue2.py", "comfy_extras/nodes_audio.py",
    "comfy_extras/nodes_lt.py", "comfy_extras/nodes_lt_audio.py",
    "comfy_extras/nodes_custom_sampler.py", "comfy_extras/nodes_primitive.py",
    "comfy_extras/nodes_video.py",
]
TEMPLATES = ROOT / ".venv-comfy/lib/python3.13/site-packages/comfyui_workflow_templates_json/templates"


def keywords(call):
    return {item.arg: item.value for item in call.keywords}


def constant(node, fallback=None):
    return node.value if isinstance(node, ast.Constant) else fallback


def type_name(node):
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Attribute):
        return node.attr.upper()
    return "COMBO"  # runtime-populated filename/sampler options


def schema_input(node):
    """Read just the selected native schema fields; never evaluate source code."""
    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
        return None
    if node.func.attr != "Input":
        return None
    kind = node.func.value.attr
    if kind == "MultiType":
        return schema_input(node.args[0])
    options = keywords(node)
    name = constant(node.args[0]) if node.args else constant(options.get("id"))
    optional = constant(options.get("optional"), False)
    return name, kind.upper(), optional


def read_source_schemas(needed):
    schemas = {}
    for relative in SOURCES:
        tree = ast.parse((COMFY / relative).read_text(encoding="utf-8"))
        classes = {node.name: node for node in tree.body if isinstance(node, ast.ClassDef)}
        aliases = {}
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == "NODE_CLASS_MAPPINGS"
                for target in node.targets
            ) and isinstance(node.value, ast.Dict):
                aliases = {constant(key): value.id for key, value in zip(node.value.keys, node.value.values)
                           if isinstance(value, ast.Name)}
        for class_name, cls in classes.items():
            methods = {item.name: item for item in cls.body if isinstance(item, ast.FunctionDef)}
            if "define_schema" in methods:
                schema = next((node for node in ast.walk(methods["define_schema"])
                               if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                               and node.func.attr == "Schema"), None)
                if schema is None:
                    continue
                options = keywords(schema)
                node_id = constant(options.get("node_id"))
                if node_id not in needed or not isinstance(options.get("inputs"), ast.List):
                    continue
                inputs = {}
                for item in options["inputs"].elts:
                    parsed = schema_input(item)
                    if parsed:
                        inputs[parsed[0]] = parsed[1:]
                outputs = [item.func.value.attr.upper() for item in options["outputs"].elts]
                schemas[node_id] = (inputs, outputs)
            elif "INPUT_TYPES" in methods and any(alias in needed and name == class_name for alias, name in aliases.items()):
                returned = next((node.value for node in ast.walk(methods["INPUT_TYPES"])
                                 if isinstance(node, ast.Return) and isinstance(node.value, ast.Dict)), None)
                outputs = next((node.value for node in cls.body if isinstance(node, ast.Assign)
                                and any(isinstance(target, ast.Name) and target.id == "RETURN_TYPES"
                                        for target in node.targets)), None)
                if returned is None or not isinstance(outputs, ast.Tuple):
                    continue
                inputs = {}
                for group_key, group in zip(returned.keys, returned.values):
                    if constant(group_key) not in {"required", "optional"} or not isinstance(group, ast.Dict):
                        continue
                    for key, value in zip(group.keys, group.values):
                        inputs[constant(key)] = (type_name(value.elts[0]), constant(group_key) == "optional")
                for alias, name in aliases.items():
                    if name == class_name:
                        schemas[alias] = (inputs, [type_name(item) for item in outputs.elts])
    return schemas


class WorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.payloads = {name: json.loads(path.read_text()) for name, path in RECIPES.items()}
        cls.graphs = {name: payload["graph"] for name, payload in cls.payloads.items()}

    def test_root_cli_inputs_wrap_api_graphs(self):
        for name, payload in self.payloads.items():
            with self.subTest(recipe=name):
                self.assertEqual(set(payload), {"graph"})
                self.assertIsInstance(payload["graph"], dict)
                self.assertTrue(payload["graph"])

    def test_api_graphs_are_connected_and_acyclic(self):
        for name, graph in self.graphs.items():
            with self.subTest(recipe=name):
                self.assertIsInstance(graph, dict)
                self.assertNotIn("nodes", graph)
                visited, visiting = set(), set()

                def visit(node_id):
                    self.assertNotIn(node_id, visiting, "workflow contains a cycle")
                    if node_id in visited:
                        return
                    self.assertIn(node_id, graph)
                    visiting.add(node_id)
                    node = graph[node_id]
                    self.assertIsInstance(node["class_type"], str)
                    for value in node["inputs"].values():
                        if isinstance(value, list):
                            self.assertEqual(len(value), 2)
                            self.assertIsInstance(value[1], int)
                            self.assertGreaterEqual(value[1], 0)
                            visit(value[0])
                    visiting.remove(node_id)
                    visited.add(node_id)

                for node_id, node in graph.items():
                    if node["class_type"].startswith("Save"):
                        prefix = node["inputs"]["filename_prefix"]
                        self.assertFalse(Path(prefix).is_absolute())
                        self.assertNotIn("..", Path(prefix).parts)
                        visit(node_id)
                self.assertEqual(visited, set(graph), "unused nodes in API workflow")

    @unittest.skipUnless((COMFY / "nodes.py").is_file(), "prepared ComfyUI checkout unavailable")
    def test_native_node_names_required_inputs_and_link_types(self):
        needed = {node["class_type"] for graph in self.graphs.values() for node in graph.values()}
        schemas = read_source_schemas(needed)
        for name, graph in self.graphs.items():
            for node_id, node in graph.items():
                with self.subTest(recipe=name, node=node_id, kind=node["class_type"]):
                    self.assertIn(node["class_type"], schemas)
                    inputs, outputs = schemas[node["class_type"]]
                    self.assertTrue({key for key, (_, optional) in inputs.items() if not optional}
                                    <= node["inputs"].keys(), "missing required input")
                    for key, value in node["inputs"].items():
                        # V3 dynamic combos flatten selected child fields (format.codec).
                        base_key = key.split(".", 1)[0]
                        self.assertIn(base_key, inputs, "unknown native input")
                        if isinstance(value, list):
                            upstream = graph[value[0]]
                            output_types = schemas[upstream["class_type"]][1]
                            self.assertLess(value[1], len(output_types))
                            self.assertEqual(output_types[value[1]], inputs[key][0])

    def test_manifest_retains_standalone_scripts_and_shared_tasks(self):
        profiles = {item["id"]: item for item in json.loads((ROOT / "media-runtimes.json").read_text())["profiles"]}
        for profile_id, script, tasks in [
            ("ltx-2.5", "generate-ltx-2.5.sh", ["video", "music"]),
            ("yue2-music", "generate-yue2.sh", ["music"]),
        ]:
            with self.subTest(profile=profile_id):
                profile = profiles[profile_id]
                self.assertEqual(profile["runtime"], "ComfyUI")
                self.assertNotIn("device", profile)
                self.assertEqual(profile["script"], f"media-models/{script}")
                self.assertEqual(profile["tasks"], tasks)
        self.assertIn("SaveVideo", {node["class_type"] for node in self.graphs["ltx"].values()})
        for graph in self.graphs.values():
            self.assertIn("SaveAudioAdvanced", {node["class_type"] for node in graph.values()})

    def test_ltx_preview_keeps_audio_video_timing_aligned(self):
        graph = self.graphs["ltx"]
        self.assertEqual(graph["8"]["inputs"]["length"], graph["9"]["inputs"]["frames_number"])
        self.assertEqual(graph["8"]["inputs"]["length"] % 8, 1)
        self.assertEqual(graph["7"]["inputs"]["frame_rate"], graph["9"]["inputs"]["frame_rate"])
        self.assertEqual(graph["9"]["inputs"]["frame_rate"], graph["19"]["inputs"]["fps"])

    @unittest.skipUnless(TEMPLATES.is_dir(), "installed official workflow templates unavailable")
    def test_recipes_follow_installed_official_models_nodes_and_schedule(self):
        for recipe, template in [("yue2", "audio_yue2_text2music.json"), ("ltx", "video_ltx2_5_t2v.json")]:
            with self.subTest(recipe=recipe):
                official = json.loads((TEMPLATES / template).read_text())
                nodes = [node for graph in [official, *official["definitions"]["subgraphs"]]
                         for node in graph["nodes"]]
                official_types = {node["type"] for node in nodes}
                # The LTX preview adds an explicit audio saver to the official AV graph.
                self.assertTrue({node["class_type"] for node in self.graphs[recipe].values()}
                                <= official_types | {"SaveAudioAdvanced"})
                official_widgets = [value for node in nodes for value in (node.get("widgets_values") or [])
                                    if isinstance(value, str)]
                for node in self.graphs[recipe].values():
                    for key in ("ckpt_name", "unet_name", "clip_name", "vae_name"):
                        if key in node["inputs"]:
                            self.assertIn(node["inputs"][key], official_widgets)
                    if node["class_type"] == "ManualSigmas":
                        self.assertIn(node["inputs"]["sigmas"], official_widgets)


if __name__ == "__main__":
    unittest.main()
