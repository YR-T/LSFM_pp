#@ File(label="Input TIFF directory", style="directory") inputDir
#@ File(label="Output TrackMate XML directory", style="directory") outputDir
#@ Double(label="LoG radius in pixels", value=2.5) radius
#@ Double(label="LoG quality threshold", value=150.0) threshold
#@ Boolean(label="Median filter before detection", value=false) medianFilter
#@ Boolean(label="Overwrite existing XML", value=false) overwrite

import ij.IJ

import fiji.plugin.trackmate.Logger
import fiji.plugin.trackmate.Model
import fiji.plugin.trackmate.Settings
import fiji.plugin.trackmate.TrackMate
import fiji.plugin.trackmate.detection.LogDetectorFactory
import fiji.plugin.trackmate.io.TmXmlWriter
import fiji.plugin.trackmate.tracking.jaqaman.SparseLAPTrackerFactory


def images = inputDir.listFiles()
    .findAll { file ->
        def lower = file.getName().toLowerCase()
        file.isFile() && (lower.endsWith(".tif") || lower.endsWith(".tiff"))
    }
    .sort { first, second -> first.getName() <=> second.getName() }

if (images.isEmpty()) {
    throw new RuntimeException("No TIFF files found in " + inputDir)
}
outputDir.mkdirs()

def radiusTag = String.format(java.util.Locale.US, "%s", radius)
    .replaceAll(/\.0$/, "")
    .replace(".", "p")
def medianTag = medianFilter ? "on" : "off"
def suffix = "_log_r" + radiusTag + "_q150_median_" + medianTag + ".xml"

images.eachWithIndex { imageFile, index ->
    def stem = imageFile.getName().replaceFirst(/(?i)\.tiff?$/, "")
    def outputFile = new File(outputDir, stem + suffix)
    if (outputFile.isFile() && outputFile.length() > 0 && !overwrite) {
        println(
            "BATCH_PROGRESS=" + (index + 1) + "/" + images.size()
            + "|CACHED|" + imageFile.getName()
        )
        return
    }

    def imp = IJ.openImage(imageFile.getAbsolutePath())
    if (imp == null) {
        throw new RuntimeException("Could not open input image: " + imageFile)
    }
    try {
        def calibration = imp.getCalibration()
        calibration.pixelWidth = 1.0
        calibration.pixelHeight = 1.0
        calibration.pixelDepth = 1.0
        calibration.frameInterval = 1.0
        calibration.setUnit("pixel")
        calibration.setTimeUnit("frame")
        imp.setCalibration(calibration)

        def model = new Model()
        model.setLogger(Logger.VOID_LOGGER)
        model.setPhysicalUnits("pixel", "frame")

        def settings = new Settings(imp)
        settings.detectorFactory = new LogDetectorFactory()
        settings.detectorSettings = settings.detectorFactory.getDefaultSettings()
        settings.detectorSettings["TARGET_CHANNEL"] = 1
        settings.detectorSettings["RADIUS"] = radius
        settings.detectorSettings["THRESHOLD"] = threshold
        settings.detectorSettings["DO_SUBPIXEL_LOCALIZATION"] = true
        settings.detectorSettings["DO_MEDIAN_FILTERING"] = medianFilter
        settings.trackerFactory = new SparseLAPTrackerFactory()
        settings.trackerSettings = settings.trackerFactory.getDefaultSettings()
        settings.addAllAnalyzers()

        def trackmate = new TrackMate(model, settings)
        if (!trackmate.checkInput()) {
            throw new RuntimeException(trackmate.getErrorMessage())
        }
        if (!trackmate.process()) {
            throw new RuntimeException(trackmate.getErrorMessage())
        }

        def temporary = new File(outputFile.getAbsolutePath() + ".tmp")
        def writer = new TmXmlWriter(temporary)
        writer.appendModel(model)
        writer.appendSettings(settings)
        writer.writeToFile()
        if (outputFile.exists() && !outputFile.delete()) {
            throw new RuntimeException("Could not replace " + outputFile)
        }
        if (!temporary.renameTo(outputFile)) {
            throw new RuntimeException("Could not rename temporary XML: " + temporary)
        }
        println(
            "BATCH_PROGRESS=" + (index + 1) + "/" + images.size()
            + "|SPOTS=" + model.getSpots().getNSpots(true)
            + "|" + imageFile.getName()
        )
    } finally {
        imp.close()
    }
}

println("BATCH_COMPLETE=" + images.size())
