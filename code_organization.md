# io
This modules is to deal with loading and saving data.

- load text files
- convert to our data structure
- export our data to other formats like npy or hdf5


generates two separate datasets:

- Tail data
	-	contains only the data related to the changes in tail over time
- Centroid data
	-	contains only the data related to the changes in fish position over time

*Could also add other datasets like eye / stimulus*

<span style="color:gray"> *think about a way to define the content of the structures to allow the users to expand with their data sources* </span>


# plotting
This is where the all code related to plots goes.
This includes things like colormaps and plot settings.

Assumes data is always in our data structure.

If necessary we can create hot swappable plots.


# utils
- min/max filters
- mike's boxcar filter
- matrix rotations
- interpolate data


# error correction
has code to run basic fixes on `dataset` data.
fixes things like head orientation bugs


# segmentation
Takes a `dataset` object and produces a `bouts` object.
Responsible for segmenting the data into bouts.
Each type of segmentation algorithm implement its extra functions.


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
bouts = algorithm.segment_data(data)

```
*corrections related to specific algorithms i.e. how joao tags broken bouts must be implemented at the inherited class level instead of the base class.*




# classification
Takes a `bout` object and produces a `bout class` object.

```python
class BoutClassificationAlgorithm(ABC):

	@abstractmethod
	def classify_data(self.segmented_data):
		# process segmented_data
		return classified_bouts


class MarquesBoutClassification(BoutClassificationAlgorithm):
	def classify_data(self.segmented_data):
		# process segmented_data
		return classified_bouts

...
#use case
from megabouts.classification import MarquesBoutClassification
algorithm = MarquesBoutClassification(some,params)
bout_classes = algorithm.classify_data(bouts)
```


# pipelines
This is where we put aggregator pipelines that use the most the other functions. 

They take a `dataset` object and output a `boutclass` object

- Marques et al.
- Trajectory Only
- HR Only

*different pipelines for different freqs*

loading the data is not part of the pipeline since it is easier for external data formats.

`load_data > pre_process > bout_segmentation > bout_classification`


```python
from megabouts.io import load_text
from megabouts.pipelines import marques_et_al

dataset = load_text(path)
# runs both segmentation and classification with joao's method
bouts = marques_et_al(dataset,param1,param2)
```

# stuff I don't know where to put


# ideas


```python
import megabouts.io as mio
from megabouts.segmentation import ClassicalSegmentation as segment

dataset = mio.loadFromText(path)

bouts = seg(dataset)

classes = model.predict(bouts)

```

# questions


# Marques et al.

![MATLAB dependencies](https://yuml.me/eb7562d4.svg "MATLAB dependencies")



| MATLAB										| Python        |
| --------------------------------------------- |---------------|
| BeatDetector 									| segmentation |
| BeatMaxAngularSpeedCorr 						| segmentation |
| BeatParametersCalculator 						| segmentation |
| boutCatFunctionWithDistanceToCenterCalculator | classification |
| BoutDetectorCurvatureFunction 				| segmentation |
| BoutParametersCalculator 						| segmentation |
| BoutParametersCalculatorMaxAngularSpeed 		| segmentation |
| boxcarf 										| --- |
| C1C2Calculator 								| segmentation |
| calculateDistance 							| utils |
| EnumeratorBeatKinPar 							| --- |
| EnumeratorBoutKinPar 							| --- |
| findPeaks 									| preprocessing |
| FishAngleCalculatorFast 						| preprocessing |
| FixHeadDirErrorGaps 							| preprocessing |
| FixTailTrackingGaps 							| preprocessing
| halfBeatTailWaveExtrapolation 				| segmentation |
| maxFilterFast 								| utils.filters |
| MikeKinAnalysisAndBoutCat						| pipeline |
| mikesmoothv 									| --- |
| mikesmoothv2 									| utils.filters |
| minFilterFast 								| utils.filters |
| ObjDetector 									| segmentation |
| putInJoaosSpace								| classification |
| rotate_matrix 								| utils.math |
| speedWithDirectionCalculator 					| segmentation |
| TailCurvatureCalculator 						| segmentation |
| yawCalculator 								| preprocessing |