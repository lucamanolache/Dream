{
  description = "Dev environment using uv with CUDA-enabled PyTorch";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    cachix.url = "github:cachix/cachix";
  };

  nixConfig = {
    extra-substituters = [ "https://nix-community.cachix.org" ];
    extra-trusted-public-keys = [
      "nix-community.cachix.org-1:mB9FSh9qf2dCimDSUo8Zy7bkq5CX+/rkCWyvRCYg3Fs="
    ];
  };

  outputs = { self, nixpkgs, cachix, ... }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs {
        inherit system;
        config.allowUnfree = true;
        config.cudaSupport = true;
      };
    in {
      devShells.${system}.default = pkgs.mkShell {
        packages = [
          pkgs.tmux
          pkgs.uv
          pkgs.python311
          pkgs.python311Packages.setuptools
          pkgs.python311Packages.wheel
          pkgs.python311Packages.pip
          pkgs.cudatoolkit
          pkgs.cudaPackages.cudnn
          pkgs.cudaPackages.cuda_cudart
          pkgs.gcc13
          pkgs.gcc13.cc.lib
          pkgs.stdenv.cc.cc.lib
          pkgs.pkg-config
          cachix.packages.${system}.default
        ];

        shellHook = ''
          # Setup CUDA environment
          export CUDA_PATH=${pkgs.cudatoolkit}
          export CC=${pkgs.gcc13}/bin/gcc
          export CXX=${pkgs.gcc13}/bin/g++
          export PATH=${pkgs.gcc13}/bin:$PATH
          export LD_LIBRARY_PATH=${pkgs.lib.makeLibraryPath [
            "/run/opengl-driver"
            pkgs.cudatoolkit
            pkgs.cudaPackages.cudnn
            pkgs.gcc13.cc.lib
            pkgs.stdenv.cc.cc.lib
          ]}:$LD_LIBRARY_PATH
          export LIBRARY_PATH=${pkgs.lib.makeLibraryPath [
            pkgs.cudatoolkit
            pkgs.gcc13.cc.lib
            pkgs.stdenv.cc.cc.lib
          ]}:$LIBRARY_PATH
          
          # Setup Python development environment
          export PYTHONPATH=${pkgs.python311Packages.setuptools}/lib/python3.11/site-packages:$PYTHONPATH
          export CPATH=${pkgs.python311}/include/python3.11:${pkgs.python311}/include/python3.11/internal:$CPATH
          export PKG_CONFIG_PATH=${pkgs.pkg-config}/lib/pkgconfig:$PKG_CONFIG_PATH

          # Setup uv-managed venv with Python 3.11
          if [ ! -d .venv ]; then
            uv venv .venv --python ${pkgs.python311}/bin/python
          fi

          uv pip install -r requirements.txt

          source .venv/bin/activate
        '';
      };
    };
}

