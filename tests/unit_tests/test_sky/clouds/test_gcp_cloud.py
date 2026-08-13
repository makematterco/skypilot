"""Test the GCP class."""

import pytest

from sky.clouds import gcp as gcp_mod


class TestGetGpuImageId:
    """Tests for GCP._get_gpu_image_id GPU image selection."""

    @pytest.mark.parametrize('acc_name',
                             ['T4', 'A100', 'A100-80GB', 'L4', 'H100', 'B200'])
    def test_turing_and_later_uses_cuda13_default(self, acc_name):
        assert gcp_mod.GCP._get_gpu_image_id(
            acc_name) == gcp_mod._DEFAULT_GPU_IMAGE_ID

    @pytest.mark.parametrize('acc_name', ['V100', 'P100', 'P4', 'M60'])
    def test_pre_turing_uses_legacy_cuda12(self, acc_name):
        assert gcp_mod.GCP._get_gpu_image_id(
            acc_name) == gcp_mod._DEFAULT_GPU_CUDA12_IMAGE_ID

    def test_k80_uses_k80_image(self):
        assert gcp_mod.GCP._get_gpu_image_id(
            'K80') == gcp_mod._DEFAULT_GPU_K80_IMAGE_ID


class TestGetGpuAccType:
    """Tests for GCP._get_gpu_acc_type acceleratorType naming."""

    @pytest.mark.parametrize(('acc_name', 'expected'), [
        ('A100-80GB', 'nvidia-a100-80gb'),
        ('L4', 'nvidia-l4'),
        ('B200', 'nvidia-b200'),
        ('H100', 'nvidia-h100-80gb'),
        ('H100-MEGA', 'nvidia-h100-mega-80gb'),
        ('H200', 'nvidia-h200-141gb'),
    ])
    def test_special_cased_names(self, acc_name, expected):
        assert gcp_mod.GCP._get_gpu_acc_type(acc_name) == expected

    @pytest.mark.parametrize(('acc_name', 'expected'), [
        ('T4', 'nvidia-tesla-t4'),
        ('A100', 'nvidia-tesla-a100'),
        ('V100', 'nvidia-tesla-v100'),
    ])
    def test_legacy_tesla_fallback(self, acc_name, expected):
        assert gcp_mod.GCP._get_gpu_acc_type(acc_name) == expected

    def test_rtxpro6000_does_not_use_tesla_fallback(self):
        """G4's RTX PRO 6000 must not get the 'nvidia-tesla-*' name.

        'nvidia-tesla-rtxpro6000' does not exist in GCP; requesting it fails
        with notFound, which SkyPilot surfaces as a spurious
        ResourcesUnavailableError. The real type is 'nvidia-rtx-pro-6000'.
        """
        assert gcp_mod.GCP._get_gpu_acc_type(
            'RTXPRO6000') == 'nvidia-rtx-pro-6000'


class TestG4AcceleratorCatalog:
    """G4 / RTX PRO 6000 are bundled with the machine type, not attachable."""

    @pytest.mark.parametrize(('instance_type', 'acc_count'), [
        ('g4-standard-6', 1),
        ('g4-standard-12', 1),
        ('g4-standard-24', 1),
        ('g4-standard-48', 1),
        ('g4-standard-96', 2),
        ('g4-standard-192', 4),
        ('g4-standard-384', 8),
    ])
    def test_g4_instance_types_report_their_bundled_gpus(
            self, instance_type, acc_count):
        """Without this, g4 shapes look CPU-only and GPU requests fail."""
        from sky.catalog import gcp_catalog
        assert gcp_catalog.get_accelerators_from_instance_type(
            instance_type) == {
                'RTXPRO6000': acc_count
            }

    def test_cpu_instance_types_have_no_bundled_gpus(self):
        from sky.catalog import gcp_catalog
        assert gcp_catalog.get_accelerators_from_instance_type(
            'n2-standard-8') is None
