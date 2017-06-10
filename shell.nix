with import <nixpkgs> {};
stdenv.mkDerivation {
  name = "starbox";
  nativeBuildInputs = with pkgs; [ rustChannels.stable.rust pkgconfig ];
  buildInputs = with pkgs; [ ilmbase openexr ];
}
