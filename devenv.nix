{ pkgs, ... }:

{
  packages = with pkgs; [
    autoconf
    automake
    cmake
    curl
    squashfsTools
    pkg-config
    libmnl
    inotify-tools
    git-lfs
    rustup
    onnxruntime
  ];

  languages = {
    elixir = {
      enable = true;
      package = pkgs.beam28Packages.elixir_1_18;
    };
    erlang = {
      enable = true;
      package = pkgs.beam28Packages.erlang;
    };
    rust.enable = true;
  };

  cachix = {
    enable = true;
    pull = [ "devenv" "pre-commit-hooks" ];
  };

  env.ORT_DYLIB_PATH = "${pkgs.lib.getLib pkgs.onnxruntime}/lib/libonnxruntime.so";
}
