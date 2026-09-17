import unittest

from coral.benchmarking.hardware import (
    get_device,
    model_load_kwargs,
    move_model_to_device,
    should_compile,
)


class _Availability:
    def __init__(self, available):
        self._available = available

    def is_available(self):
        return self._available


class _Backends:
    def __init__(self, mps_available):
        self.mps = _Availability(mps_available)


class _FakeTorch:
    float16 = "float16"

    def __init__(self, cuda_available=False, mps_available=False):
        self.cuda = _Availability(cuda_available)
        self.backends = _Backends(mps_available)

    @staticmethod
    def device(device_type):
        return FakeDevice(device_type)


class FakeDevice:
    def __init__(self, device_type):
        self.type = device_type

    def __eq__(self, other):
        return isinstance(other, FakeDevice) and self.type == other.type


class _FakeModel:
    def __init__(self):
        self.device = None

    def to(self, device):
        self.device = device
        return self


class HardwareTests(unittest.TestCase):
    def test_get_device_prefers_cuda_when_both_accelerators_are_available(self):
        torch = _FakeTorch(cuda_available=True, mps_available=True)

        self.assertEqual(get_device(torch), FakeDevice("cuda"))

    def test_get_device_uses_mps_when_cuda_is_unavailable(self):
        torch = _FakeTorch(mps_available=True)

        self.assertEqual(get_device(torch), FakeDevice("mps"))

    def test_get_device_falls_back_to_cpu(self):
        torch = _FakeTorch()

        self.assertEqual(get_device(torch), FakeDevice("cpu"))

    def test_mps_loads_half_precision_without_cuda_quantization(self):
        torch = _FakeTorch(mps_available=True)

        kwargs = model_load_kwargs(
            torch,
            FakeDevice("mps"),
            device_map="auto",
            load_in_8bit=True,
        )

        self.assertEqual(
            kwargs,
            {"low_cpu_mem_usage": True, "torch_dtype": "float16"},
        )

    def test_cuda_preserves_automatic_placement_and_8_bit_loading(self):
        torch = _FakeTorch(cuda_available=True)

        kwargs = model_load_kwargs(torch, FakeDevice("cuda"))

        self.assertEqual(kwargs, {"device_map": "auto", "load_in_8bit": True})

    def test_cuda_honors_legacy_loading_overrides(self):
        torch = _FakeTorch(cuda_available=True)

        kwargs = model_load_kwargs(
            torch,
            FakeDevice("cuda"),
            device_map="cuda:0",
            load_in_8bit=False,
        )

        self.assertEqual(kwargs, {"device_map": "cuda:0", "load_in_8bit": False})

    def test_cpu_loading_uses_no_accelerator_specific_options(self):
        torch = _FakeTorch()

        self.assertEqual(model_load_kwargs(torch, FakeDevice("cpu")), {})

    def test_mps_model_is_moved_to_the_selected_device(self):
        model = _FakeModel()
        device = FakeDevice("mps")

        returned_model = move_model_to_device(model, device)

        self.assertIs(returned_model, model)
        self.assertEqual(model.device, device)

    def test_cuda_model_placement_is_left_to_device_map(self):
        model = _FakeModel()

        move_model_to_device(model, FakeDevice("cuda"))

        self.assertIsNone(model.device)

    def test_torch_compile_is_disabled_for_mps(self):
        self.assertFalse(should_compile(FakeDevice("mps"), "darwin"))

    def test_torch_compile_remains_enabled_for_cuda_on_unix(self):
        self.assertTrue(should_compile(FakeDevice("cuda"), "linux"))


if __name__ == "__main__":
    unittest.main()
