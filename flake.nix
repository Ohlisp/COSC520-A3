{
  description = "Graph approximation algorithms and unit tests";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";

  outputs = { nixpkgs, ... }:
    let
      systems = [ "aarch64-darwin" "x86_64-darwin" "aarch64-linux" "x86_64-linux" ];
    in {
      devShells = nixpkgs.lib.genAttrs systems (system:
        let
          pkgs = nixpkgs.legacyPackages.${system};
          python = pkgs.python3.withPackages (ps: [ ps.pytest ps.matplotlib ]);
          benchmark = pkgs.writeShellScriptBin "run-benchmark" ''
            exec ${python}/bin/python -m src.run_benchmark "$@"
          '';
        in {
          default = pkgs.mkShell {
            packages = [ python benchmark ];
          };
        });
    };
}
