# io
This modules is to deal with loading and saving data.

- load text files
- convert to our data structure
- export our data to other formats like npy or hdf5

# plotting
This is where the all code related to plots goes.
This includes things like colormaps and plot settings.

Assumes data is always in our data structure.

If necessary we can create hot swappable plots.


# utils
- min/max filters
- mike's boxcar filter
- matrix rotations


# segmentation
Takes a `dataset` object and produces a `bouts` object.
Responsible for segmenting the data into bouts.
Each type of segmentation algorithm implement its extra functions.

```python
class BoutSegmentationAlgorithm(ABC):

	@abstractmethod
	def segment_data(raw_data):
		# process raw_data
		return bouts


class MarquesBoutSegmentation(BoutSegmentationAlgorithm):
	def segment_data(raw_data):
		# process raw_data
		return bouts

...
#use case
from megabouts.segmentation import MarquesBoutSegmentation
MarquesBoutSegmentation.segment_data(data)

```


```python
class BoutSegmentationAlgorithm(ABC):

	@abstractmethod
	def segment_data(self.raw_data):
		# process raw_data
		return bouts


class MarquesBoutSegmentation(BoutSegmentationAlgorithm):
	def segment_data(self.raw_data):
		# process raw_data
		return bouts

...
#use case
from megabouts.segmentation import MarquesBoutSegmentation
algorithm = MarquesBoutSegmentation(some,params)
algorithm.segment_data(data)

```



# classification
Takes a `bout` object and produces a `bout class` object.


# pipelines
This is where we put aggregator pipelines that use the most the other functions. 

They take a `dataset` object and output a `boutclass` object

- Marques et al.
- Trajectory Only
- HR Only

```python
from megabouts.io import load_text
from megabouts.pipelines import marques_et_al

data = load_text(path)
# runs both segmentation and classification with joao's method
bouts = marques_et_al(dataset)
```

# stuff I don't know where to put
- correcting tracking errors
- remove double bouts
- remove broken bouts

# ideas
- create skeleton class for data and allow people to expand that with their data formats

```python
import megabouts.io as mio
from megabouts.segmentation import ClassicalSegmentation as segment

dataset = mio.loadFromText(path)

bouts = seg(dataset)

classes = model.predict(bouts)

```

# questions
- Where do we choose the framerate?
