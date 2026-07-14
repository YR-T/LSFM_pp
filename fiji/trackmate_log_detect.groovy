#@ File(label="Input TIFF", style="file") inputFile
#@ File(label="Output TrackMate XML", style="save") outputFile
#@ Double(label="LoG radius in pixels", value=3.0) radius
#@ Double(label="LoG quality threshold", value=0.0) threshold
#@ Boolean(label="Median filter before detection", value=true) medianFilter

import ij.IJ

import fiji.plugin.trackmate.Logger
import fiji.plugin.trackmate.Model
import fiji.plugin.trackmate.Settings
import fiji.plugin.trackmate.TrackMate
import fiji.plugin.trackmate.detection.LogDetectorFactory
import fiji.plugin.trackmate.io.TmXmlWriter
import fiji.plugin.trackmate.tracking.jaqaman.SparseLAPTrackerFactory


def imp = IJ.openImage(inputFile.getAbsolutePath())
if (imp == null) {
    throw new RuntimeException("Could not open input image: " + inputFile)
}

// Treat every spatial dimension as unitless pixels with spacing 1.
def calibration = imp.getCalibration()
calibration.pixelWidth = 1.0
calibration.pixelHeight = 1.0
calibration.pixelDepth = 1.0
calibration.frameInterval = 1.0
calibration.setUnit("pixel")
calibration.setTimeUnit("frame")
imp.setCalibration(calibration)

def model = new Model()
model.setLogger(Logger.IJ_LOGGER)
model.setPhysicalUnits("pixel", "frame")

def settings = new Settings(imp)
settings.detectorFactory = new LogDetectorFactory()
settings.detectorSettings = settings.detectorFactory.getDefaultSettings()
settings.detectorSettings["TARGET_CHANNEL"] = 1
settings.detectorSettings["RADIUS"] = radius
settings.detectorSettings["THRESHOLD"] = threshold
settings.detectorSettings["DO_SUBPIXEL_LOCALIZATION"] = true
settings.detectorSettings["DO_MEDIAN_FILTERING"] = medianFilter

// A tracker is configured so TrackMate can run its standard process pipeline.
// For a single 2D image there are no temporal links to create.
settings.trackerFactory = new SparseLAPTrackerFactory()
settings.trackerSettings = settings.trackerFactory.getDefaultSettings()

// Includes intensity, contrast, SNR, morphology, and other available features.
settings.addAllAnalyzers()

def trackmate = new TrackMate(model, settings)
if (!trackmate.checkInput()) {
    throw new RuntimeException(trackmate.getErrorMessage())
}
if (!trackmate.process()) {
    throw new RuntimeException(trackmate.getErrorMessage())
}

def parent = outputFile.getParentFile()
if (parent != null) {
    parent.mkdirs()
}

def writer = new TmXmlWriter(outputFile)
writer.appendModel(model)
writer.appendSettings(settings)
writer.writeToFile()

println("TRACKMATE_INPUT=" + inputFile.getAbsolutePath())
println("TRACKMATE_OUTPUT=" + outputFile.getAbsolutePath())
println("TRACKMATE_SPOTS=" + model.getSpots().getNSpots(true))
println("TRACKMATE_RADIUS=" + radius)
println("TRACKMATE_THRESHOLD=" + threshold)
println("TRACKMATE_MEDIAN_FILTER=" + medianFilter)

imp.close()
