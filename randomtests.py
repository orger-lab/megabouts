from megabouts.segmentation.marques_segmentation import MarquesSegmenter

algorithm = MarquesSegmenter()
bouts = algorithm.segment("some data")
print(bouts)