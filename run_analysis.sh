testing="$1"

zip -j tools ./src/tools/*
zip configuration ./src/configuration/*

if [ "$testing" == "-t" ]; then
    python -m unittest discover tests/
else
    spark-submit src/main.py --python-files tools.zip configuration.zip
fi