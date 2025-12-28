{
  inputs.nixpkgs.url = "nixpkgs/release-25.11";
  outputs = {
    self,
    nixpkgs,
  }: let
    system = "x86_64-linux";
    pkgs = nixpkgs.legacyPackages.${system};
    fhs = pkgs.buildFHSEnv {
      name = "fhs-shell";
      targetPkgs = pkgs: with pkgs; [uv typst gnumake zlib];
    };
  in {
    devShells.${system}.default = fhs.env;
  };
}
