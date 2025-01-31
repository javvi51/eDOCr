# eDOCr
## See the version 2 of eDOCr in its new repo [edocr2](https://github.com/javvi51/edocr2)
eDOCr is a packaged version of [keras-ocr](https://github.com/faustomorales/keras-ocr) that facilitates end-to-end digitization of mechanical EDs. Developed for Windows OS and using Python as the primary programming language. The implementation is discussed in the researh paper [Optical character recognition on engineering drawings to achieve automation in production quality control](https://www.frontiersin.org/articles/10.3389/fmtec.2023.1154132/full) 

## Getting Started

### Installation

`eDOCr` supports Python >= 3.6 and TensorFlow >= 2.0.0. The test environment I used is Python = 3.9 and Tensorflow = 2.10.1.
Install your prefered distribution platform, [Anaconda](https://www.anaconda.com/products/distribution) is recommended.
Open Anaconda Prompt and type the following commands:


```bash
conda create -n edocr python=3.9 -y 
conda activate edocr

# To install from PyPi
conda install pip
pip install eDOCr

# To install from Source
cd path/to/your/folder
git clone https://github.com/javvi51/eDOCr
cd eDOCr
pip install -r requirements.txt
pip install .
```

### Using 

There are two ways of using eDOCr: from terminal and from your own python file.

#### From Terminal:

We need to locate the ocr_it.py file in our system. If you have installed using pip, it will probably come in ```C:\Users\YOUR_USER\.conda\envs\edocr\Lib\site-packages\eDOCr\ocr_it.py```. If you have installed from source, it will be at your selected folder. All you need to do is:

```bash
python PATH/TO/YOUR/FOLDER/eDOCr/ocr_it.py PATH/TO/YOUR/DRAWING/my_drawing.pdf

```

Additional commands you can use are:

```bash
# Specify the destination path. By default, it is the path you are running your code from.
--dest-folder PATH/TO/YOUR/DESTINATION/FOLDER
# Does the drawing have watermark you want to remove? By default, it is not considered.
--water
# Advance Setting: Set a custom threshold distance (in px.) for grouping detections. Default is 20px.
--cluster 25
```

#### From your own python file

More customization is possible using your own python file, such as selecting a different model, alphabet or changing colors.

```bash
# Importing packages
import os
from eDOCr import tools
import cv2
import string
from skimage import io

# Loading image and destination file
dest_DIR = 'tests/test_Results'
file_path = 'tests/test_samples/Candle_holder.jpg'
filename = os.path.splitext(os.path.basename(file_path))[0]
img = cv2.imread(file_path)

# Selecting alphabet and model (Note that alphabet and alphabet model need to match)
GDT_symbols = '⏤⏥○⌭⌒⌓⏊∠⫽⌯⌖◎↗⌰'
FCF_symbols = 'ⒺⒻⓁⓂⓅⓈⓉⓊ'
Extra = '(),.+-±:/°"⌀'

alphabet_dimensions = string.digits + 'AaBCDRGHhMmnx' + Extra
model_dimensions = 'eDOCr/keras_ocr_models/models/recognizer_dimensions.h5'
alphabet_infoblock = string.digits+string.ascii_letters+',.:-/'
model_infoblock = 'eDOCr/keras_ocr_models/models/recognizer_infoblock.h5'
alphabet_gdts = string.digits + ',.⌀ABCD' + GDT_symbols
model_gdts = 'eDOCr/keras_ocr_models/models/recognizer_gdts.h5'

# Selecting personalized color palette and cluster setting
color_palette = {'infoblock': (180, 220, 250), 'gdts': (94, 204, 243), 'dimensions': (93, 206, 175), 'frame': (167, 234, 82), 'flag': (241, 65, 36)}
cluster_t = 20

# eDOCr functions
class_list, img_boxes = tools.box_tree.findrect(img)
boxes_infoblock, gdt_boxes, cl_frame, process_img = tools.img_process.process_rect(class_list, img)
io.imsave(os.path.join(dest_DIR, filename + '_process.jpg'), process_img)

infoblock_dict = tools.pipeline_infoblock.read_infoblocks(boxes_infoblock, img, alphabet_infoblock, model_infoblock)
gdt_dict = tools.pipeline_gdts.read_gdtbox1(gdt_boxes, alphabet_gdts, model_gdts, alphabet_dimensions, model_dimensions)
 
process_img = os.path.join(dest_DIR, filename + '_process.jpg')

dimension_dict = tools.pipeline_dimensions.read_dimensions(process_img, alphabet_dimensions, model_dimensions, cluster_t)
mask_img = tools.output.mask_the_drawing(img, infoblock_dict, gdt_dict, dimension_dict, cl_frame, color_palette)

# Record the results
io.imsave(os.path.join(dest_DIR, filename + '_boxes.jpg'), img_boxes)
io.imsave(os.path.join(dest_DIR, filename + '_mask.jpg'), mask_img)
tools.output.record_data(dest_DIR, filename, infoblock_dict, gdt_dict, dimension_dict)
```
![example of labeled image](https://github.com/javvi51/eDOCr/releases/download/v1.0.0/Candle_holder_mask.jpg)
### Training a model on a custom alphabet
Fonts are not loaded if installing from pip. To train new models, please install from source.

To train a model in a custom alphabet, a python file is provided, so that the only steps needed are:


```bash
# Importing Packages
import os
import string
from eDOCr import keras_ocr
from eDOCr.keras_ocr_models import train_recognizer

# Fixing paths and alphabet
DIR = os.getcwd()
recognizer_basepath = os.path.join(DIR, 'eDOCr/Keras_OCR_models/models')
data_dir = './tests'
alphabet = string.digits + 'AaBCDRGHhMmnx' + '().,+-±:/°"⌀'

# Number of autogenerated samples
samples = 10000

# Load white backgrounds and fonts
backgrounds = []
for i in range(0, samples):
    backgrounds.append(os.path.join('./eDOCr/Keras_OCR_models/backgrounds/0.jpg'))

fonts = []
for i in os.listdir(os.path.join(DIR, 'eDOCr/Keras_OCR_models/fonts')):
    fonts.append(os.path.join('./eDOCr/Keras_OCR_models/fonts', i))

# Choose a pretrained model if you like

pretrained_model = None 
#pretrained_model = os.path.join(recognizer_basepath,'recognizer_dimensions.h5')

# Start Training 
train_recognizer.generate_n_train(alphabet, backgrounds, fonts, recognizer_basepath=recognizer_basepath, pretrained_model=pretrained_model)
```

This work was supported by Vinnova (Sweden) under grant 2021-02481, iPROD project. The authors would like to thank Vinnova for making this research project possible.
