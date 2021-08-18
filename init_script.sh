echo "Building a conda environment"

conda config --set always_yes yes
conda create --name fr-teamc python=3.7
conda activate fr-teamc
pip install -r requirements.txt

echo "Installing the Backend for Retinaface"


cd pipeline
git clone https://github.com/StanislasBertrand/RetinaFace-tf2.git 
cd RetinaFace-tf2
make
python detect.py --sample_img="./sample-images/WC_FR.jpeg"
cd ../weights
gdown --id 1nuLihFS61FCGotF2KRcCzOqLhrr6wPAW
gdown --id 1atHsxw9XE1oxeipr008EkImy5n6-K-NR
gdown --id 1X5c_SGcOEhfrSjvGaIYqtQk0J0sG80xu
gdown --id 1YPrAuQ1_CpVhloXXXa8QuTrFk5KE76Id

conda deactivate
