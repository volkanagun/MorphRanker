package morphology.data

import java.util.Locale

class Params() {
  var locale = new Locale("tr")
  var model = "gmm"
  var seed = 17
  val threadSize = 24
  val binaryFolder = "resources/binary/"
  val modelFolder = "resources/models/"
  val tokenizerFilename = "resources/binary/dictionary.bin"
  var maxSentenceLength = 200
  var maxSentences = 10000
  var maxFeatures = 10000
  var maxLabels = 5000
  var maxRankIters = 3
  val maxWordSamples = 100
  var maxTokenWindow = 7
  var maxLabelWindow = 20
  var forwardBackward = true

  var hiddenSize = 100
  var nHeads = 4
  var batchSize = 64
  var epocs = 100
  var lrate = 0.01

  var kThreshold = 0.05
  var kThresholdArray = Array(/*0.01, 0.05, */0.1, 0.5)
  var kDims = 10
  val kDimArray = Array(10, 20, 50, 100, 200, 500)
  var tagSamples = 10000
  val minSampleArray = Array(/*1000, 5000, 10000, 30000,*/ 100000)



  val tokenSplit = ">>>"

  var sentenceFilename = "resources/text/sentences-tr.txt";
  var sentenceFolder = "resources/sentences/";
  var sentenceTrainFolder = "resources/training/"
  val sentenceLabelFilename = "resources/text/sentences-label-tr-v1.txt";
  val sentenceLabel2Filename = "resources/text/sentences-label-tr-v2.txt";
  val sentenceLabel3Filename = "resources/text/sentences-label-tr-v3.txt";
  val sentenceLabel4Filename = "resources/text/sentences-label-tr-v4.txt";
  val ambiguousLabelFilename = "resources/text/ambiguous-label-tr.txt";
  var trainingFilename = "resources/sentences/allbig.txt";
  val test2006Filename = "resources/morphology/trmor2006.test";
  val test2016Filename = "resources/morphology/trmor2016.test";
  val test2018Filename = "resources/morphology/trmor2018.gold";

  val testFilenames = Array(/*test2006Filename, test2016Filename*/test2018Filename)
  val vocabFilenames = Array(test2006Filename, test2016Filename, test2018Filename)

  val wordSkipOrder = Array[Int](1, 2, 3, 4, 5, 6, 7)
  var ambiguityThreshold = Array(1 ,5, 10, 50, 100, 5000, 1000)
  var prunningRatios = Array(1.0, 0.5, 0.25, 0.05, 0.01)
  var maxSentenceArray = Array(1000)
  var maxWindowArray = Array(4, 6, 9)
  var maxSliceArray = Array(1, 2, 3, 4, 5)
  var skipHeadArray = Array(1, 3)

  var droupoutFreq = 100
  var prunningRatio = 0.95
  var topEdges = 5

  val oneMillion = 1000000
  val aHundred = 100000
  var skipHeadAmbiguity = 2
  var maxPairAmbiguity = 6
  var maxSentenceAmbiguity = 1
  var maxSliceNgram = 1
  var maxEpocs = 100
  var maxNeuralEpocs = 1
  var trainStats = new DataStats()

  def linkMarker(i: Int): String = {
    (if (i >= 7) "Very-Distant" else if (i >= 5) "Long-Distant" else if (i >= 2) "Distant" else if (i >= 1) "Neighbour" else "Local")
  }

 /* def linkMarker(i: Int): String = {
    val str = (if (i >= 6) "Long-Distant" else if (i >= 3) "Distant" else if (i >= 1) "Neighbour" else "Local")
    str
  }*/

  def modelID(): Int = {
    val items = Array[Int](model.hashCode, maxLabelWindow, skipHeadAmbiguity, kDims, kThreshold.hashCode(), tagSamples, maxSentences * maxEpocs, maxSliceNgram, maxTokenWindow, maxPairAmbiguity, maxSentenceAmbiguity, droupoutFreq, prunningRatio.hashCode()) ++
      ambiguityThreshold
    items.foldRight[Int](seed) { case (i, main) => {
      i + 7 * main
    }}
  }

  def datasetID(): Int = {
    val items = Array[Int](maxSentenceLength, maxWordSamples, maxSentences)
    items.foldRight[Int](seed) { case (i, main) => {
      i + 7 * main
    }}
  }

  def rankModelFilename(): String = {
    modelFolder + "rank-" + modelID() + ".bin"
  }

  def neuralModelFilename(): String = {
    modelFolder + "nn-" + modelID() + ".bin"
  }

  def toTag(label: String, value: String): String = {
    "<PARAM LABEL=\"" + label + "\" VALUE=\"" + value + "\"/>\n"
  }

  def toXML(): String = {
    "<PARAMETERS>\n" +
      toTag("MODEL", model) +
      toTag("K_THRESHOLD", kThreshold.toString) +
      toTag("K_DIM", kDims.toString) +
      toTag("MIN_SAMPLES", tagSamples.toString) +
      toTag("TAG_SLICE", maxSliceNgram.toString) +
      toTag("THREAD_SIZE", threadSize.toString) +
      toTag("SKIP_HEAD_AMBIGUITY", skipHeadAmbiguity.toString) +
      toTag("MAX_PAIR_AMBIGUITY", maxPairAmbiguity.toString) +
      toTag("MAX_SEN_AMBIGUITY", maxSentenceAmbiguity.toString) +
      toTag("MAX_TOKEN_WINDOW", maxTokenWindow.toString) +
      toTag("SECOND_ORDER", wordSkipOrder.mkString("|")) +
      toTag("MAX_RANK_ITERATIONS", maxRankIters.toString) +
      toTag("MAX_EPOCS", maxEpocs.toString) +
      toTag("MAX_SENTENCES", maxSentences.toString) +
      toTag("MAX_SENTENCE_LENGTH", maxSentenceLength.toString) +
      toTag("MAX_WORD_SAMPLES", maxWordSamples.toString) +
      toTag("PRUNNING_RATIO", prunningRatio.toString) +
      "</PARAMETERS>\n" +
      "<DATA_STATS>\n" +
      trainStats.toXML() +
      "</DATA_STATS>\n"
  }

}
