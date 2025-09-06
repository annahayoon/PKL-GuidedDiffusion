import os
import unittest


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))


class TestPhase01ProjectSetup(unittest.TestCase):
    def test_directories_exist(self):
        expected_dirs = [
            "pkl_dg",
            os.path.join("pkl_dg", "data"),
            os.path.join("pkl_dg", "physics"),
            os.path.join("pkl_dg", "models"),
            os.path.join("pkl_dg", "guidance"),
            os.path.join("pkl_dg", "baselines"),
            os.path.join("pkl_dg", "evaluation"),
            os.path.join("pkl_dg", "utils"),
            "scripts",
            "configs",
            "tests",
            "assets",
            "docs",
            "notebooks",
        ]
        for rel_path in expected_dirs:
            self.assertTrue(
                os.path.isdir(os.path.join(PROJECT_ROOT, rel_path)),
                msg=f"Missing directory: {rel_path}",
            )

    def test_init_files_exist(self):
        expected_init_files = [
            os.path.join("pkl_dg", "__init__.py"),
            os.path.join("pkl_dg", "data", "__init__.py"),
            os.path.join("pkl_dg", "physics", "__init__.py"),
            os.path.join("pkl_dg", "models", "__init__.py"),
            os.path.join("pkl_dg", "guidance", "__init__.py"),
            os.path.join("pkl_dg", "baselines", "__init__.py"),
            os.path.join("pkl_dg", "evaluation", "__init__.py"),
            os.path.join("pkl_dg", "utils", "__init__.py"),
        ]
        for rel_path in expected_init_files:
            self.assertTrue(
                os.path.isfile(os.path.join(PROJECT_ROOT, rel_path)),
                msg=f"Missing file: {rel_path}",
            )

    def test_files_exist(self):
        expected_files = [
            "README.md",
            "requirements.txt",
            "setup.py",
            os.path.join("configs", "config.yaml"),
            os.path.join("docs", "implementation", "PHASE_01_PROJECT_SETUP.md"),
        ]
        for rel_path in expected_files:
            self.assertTrue(
                os.path.isfile(os.path.join(PROJECT_ROOT, rel_path)),
                msg=f"Missing file: {rel_path}",
            )

    def test_requirements_content(self):
        req_path = os.path.join(PROJECT_ROOT, "requirements.txt")
        with open(req_path, "r", encoding="utf-8") as f:
            content = f.read()
        expected_keys = [
            "torch>=2.0.0",
            "pytorch-lightning>=2.0.0",
            "hydra-core>=1.3.0",
            "wandb>=0.15.0",
            "scikit-image>=0.21.0",
        ]
        for key in expected_keys:
            self.assertIn(key, content)

    def test_config_yaml_content(self):
        config_path = os.path.join(PROJECT_ROOT, "configs", "config.yaml")
        with open(config_path, "r", encoding="utf-8") as f:
            text = f.read()
        expected_lines = [
            "defaults:",
            "  - model: unet",
            "  - data: synthesis",
            "  - physics: microscopy",
            "  - guidance: pkl",
            "  - training: ddpm",
            "  - override hydra/launcher: basic",
            "experiment:",
            "paths:",
            "wandb:",
        ]
        for line in expected_lines:
            self.assertIn(line, text)


if __name__ == "__main__":
    unittest.main()


