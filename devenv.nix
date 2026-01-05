{ pkgs, lib, config, ... }:

{
  packages = with pkgs; [
    # System tools
    git
    ruff
    gitleaks
    p7zip  # For extracting OMI 7z archives

    # System libraries needed by Python packages
    zlib
    stdenv.cc.cc.lib  # libstdc++

    # Geographic/spatial libraries (Nix-provided for binary compat)
    gdal
    proj
    geos
  ];

  languages.python = {
    enable = true;
    package = pkgs.python312;
    uv = {
      enable = true;
      sync.enable = true;
    };
  };

  env = {
    PYTHONPATH = ".";
    DATA_DIR = "./data";
    # Ensure GDAL/GEOS are found by geopandas
    GDAL_DATA = "${pkgs.gdal}/share/gdal";
    PROJ_DATA = "${pkgs.proj}/share/proj";
    # Library path for numpy and other binary packages
    LD_LIBRARY_PATH = lib.makeLibraryPath [
      pkgs.zlib
      pkgs.stdenv.cc.cc.lib
    ];
  };

  dotenv.disableHint = true;

  scripts = {
    download-data.exec = "python scripts/download_data.py";
    build-features.exec = "python scripts/build_features.py";
    train.exec = "python scripts/train_model.py";
    predict.exec = "python scripts/predict.py";
    test.exec = "pytest tests/ -v --tb=short";
    lint.exec = "ruff check src/ tests/";
    format.exec = "ruff format src/ tests/";
    notebook.exec = "jupyter lab notebooks/";
  };

  git-hooks.hooks = {
    ruff = {
      enable = true;
      entry = "${pkgs.ruff}/bin/ruff check --fix";
      files = "\\.py$";
      excludes = [ ".devenv/" "result" ".venv/" ];
    };
    gitleaks = {
      enable = true;
      entry = "${pkgs.gitleaks}/bin/gitleaks protect --staged -v";
    };
  };

  enterShell = ''
    echo ""
    echo "=============================================="
    echo "  Italian Real Estate Demographic Risk Model"
    echo "=============================================="
    echo ""
    echo "Python: $(python --version)"
    echo "Data directory: $DATA_DIR"
    echo ""
    echo "Commands:"
    echo "  download-data   - Download OMI/ISTAT data"
    echo "  build-features  - Build feature matrix"
    echo "  train           - Train models"
    echo "  predict         - Generate predictions"
    echo "  notebook        - Launch Jupyter Lab"
    echo "  test / lint     - Quality checks"
    echo ""
  '';
}
