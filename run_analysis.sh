testing="$1"

if [ ! -f "tools.zip" ]; then
    zip -j tools tools/*
fi

if [ "$testing" == "-t" ]; then
    echo "Testing is not implemented, but wait!"
else 
    spark-submit src/main.py --python-files tools.zip
fi