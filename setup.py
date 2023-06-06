import pip
from pip._internal.commands import install
import pip
pip.main(['install','git+https://github.com/huggingface/datasets.git'])
pip.main(['install','git+https://github.com/huggingface/transformers.git'])
pip.main(['install','jiwer'])
pip.main(['install','torchaudio'])
pip.main(['install', 'librosa'])

pip.main(['install','accelerate'])
pip.main(['install','numpy=1.23.'])