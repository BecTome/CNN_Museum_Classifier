
LABELS = ['Albumen photograph', 'Bronze', 'Ceramic', 'Clay', 'Engraving',
            'Etching', 'Faience', 'Glass', 'Gold', 'Graphite',
            'Hand-colored engraving', 'Hand-colored etching', 'Iron', 'Ivory',
            'Limestone', 'Lithograph', 'Marble', 'Oil on canvas',
            'Pen and brown ink', 'Polychromed wood', 'Porcelain',
            'Silk and metal thread', 'Silver', 'Steel', 'Wood',
            'Wood engraving', 'Woodblock', 'Woodcut', 'Woven fabric']

N_CLASSES = len(LABELS)

RAW_DATA_PATH = 'input/data/raw'

TRAIN_PATH = 'input/data/train/'
VAL_PATH = 'input/data/val/'
TEST_PATH = 'input/data/test/'
OUTPUT_FOLDER = 'output/'
OUTPUT_FEATEXT_PATH = 'output/featext/'
OUTPUT_FEATEXT_DATA = 'input/data/featext/'
OUTPUT_FEATMED_DATA = 'input/data/featmed/'
OUTPUT_FEATTOP_DATA = 'input/data/feattop/'



META_PATH = 'input/metadata/'
IMG_SIZE = 256
N_CHANNELS = 3