import torch


try:
    # Try to import pre-compiled extension (from setup.py build)
    from . import _simple_knn as _C
except ImportError:
    # If not available, compile via JIT on first import
    import os
    from pathlib import Path

    def _load_extension_jit():
        """JIT compile the CUDA extension if pre-built version not available."""
        from torch.utils.cpp_extension import load

        # The .cu/.cpp sources ship inside this package directory.
        _pkg_path = Path(__file__).parent
        sources = [
            _pkg_path / "ext.cpp",
            _pkg_path / "spatial.cu",
            _pkg_path / "simple_knn.cu",
        ]

        missing = [str(f) for f in sources if not f.exists()]
        if missing:
            raise FileNotFoundError(
                f"simple-knn source files missing from {_pkg_path}: {missing}. "
                "The package appears to be installed without its CUDA sources — reinstall eden-simple-knn."
            )

        # Compilation settings
        extra_cuda_cflags = [
            "-O3",
            "--use_fast_math",
            "-std=c++17",
            "--expt-relaxed-constexpr",
        ]

        extra_cflags = ["-O3", "-std=c++17"]

        # Build directory
        cuda_ver = (
            torch.version.cuda.replace(".", "_") if torch.cuda.is_available() else "cpu"
        )
        build_dir = os.path.join(
            os.path.expanduser("~"),
            ".cache",
            "torch_extensions",
            f"simple_knn_cu{cuda_ver}",
        )

        # Create build directory if it doesn't exist
        os.makedirs(build_dir, exist_ok=True)

        is_first_build = not os.path.exists(os.path.join(build_dir, "build.ninja"))

        if is_first_build:
            print("\n" + "=" * 70)
            print("Compiling simple-knn (first time only)...")
            print("This will take 1-2 minutes.")
            print("=" * 70 + "\n")

        try:
            extension = load(
                name="simple_knn_cuda",
                sources=[str(f) for f in sources],
                extra_cflags=extra_cflags,
                extra_cuda_cflags=extra_cuda_cflags,
                extra_include_paths=[str(_pkg_path)],
                build_directory=build_dir,
                verbose=is_first_build,
                with_cuda=True,
            )

            if is_first_build:
                print("\n✓ Compilation successful! Cached for future use.\n")

            return extension

        except Exception as e:
            print("\n" + "=" * 70)
            print("ERROR: Failed to compile simple-knn")
            print("=" * 70)
            print(f"\n{e}\n")
            print("Requirements:")
            print("  - CUDA toolkit installed")
            print("  - Compatible C++ compiler (gcc 7-12)")
            print("  - PyTorch with CUDA support")
            print("=" * 70 + "\n")
            raise

    # Load via JIT
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA not available. simple-knn requires CUDA.\n"
            f"PyTorch version: {torch.__version__}"
        )

    _C = _load_extension_jit()


def distCUDA2(points):
    """
    Compute KNN distances for points using CUDA.

    Parameters
    ----------
    points: torch.Tensor
        Tensor of shape (N, 3) containing 3D points

    Returns
    -------
    torch.Tensor:
        Tensor of shape (N,) containing squared distances to nearest neighbors
    """
    return _C.distCUDA2(points)
