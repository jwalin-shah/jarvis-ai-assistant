for file in $(find internal/archive/evals/ -name "*.py"); do
    sed -i 's/from evals\./from internal.archive.evals./g' $file
    sed -i 's/sys.path.insert(0, str(PROJECT_ROOT))/sys.path.insert(0, str(PROJECT_ROOT))\n# noqa: E402/g' $file
    sed -i '/import/s/$/  # noqa: E402/' $file
done
