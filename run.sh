if [ -z "$1" ]
then
	echo "Error: Must pass number of centers as argument"
	exit 0
fi
python siftExtractor.py
python kMeans.py $1
python mergeCenters.py $1
python makeDatabase.py $1
python svmTrain.py
python predict.py $1
