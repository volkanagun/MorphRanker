package morphology.experiment

import morphology.data.{Count, Dataset, MorphRanker, Params, Result, Sentence, Word}
import morphology.experiment.MorphExperiment.tests
import morphology.ranking.{DeepELMORanker, DeepSARanker, GMMRanker, MorphPredictor, RankMorpheme}

import java.io.{File, FileOutputStream, PrintWriter}
import scala.collection.parallel.CollectionConverters.{ArrayIsParallelizable, ImmutableIterableIsParallelizable}
import scala.io.Source
import scala.util.Random

class MorphExperiment(val params: Params) {

  var result = new Result(params)


  def resultFilename(datasetFilename: String): String = {
    val datasetFid = "DataSetID=" + params.datasetID()
    val modelFid = "ModelID=" + params.modelID()
    val taskName = "Task=" + datasetFilename.substring(datasetFilename.lastIndexOf("/") + 1, datasetFilename.lastIndexOf("."))
    val fid = datasetFid + "|" + modelFid + "|" + taskName
    val filename = "resources/results/" + fid + ".xml"
    filename
  }

  def save(datasetFilename: String): Unit = {
    val filename = resultFilename(datasetFilename)
    new PrintWriter(filename) {
      println(result.toXML(datasetFilename))
    }.close()
  }

  def saveAppend(): Unit = {
    val filename = "resources/results.csv"
    val stream = new FileOutputStream(filename, true)
    new PrintWriter(stream) {
      println(result.toShortLine())
    }.close()
  }

  def exists(datasetFilename: String): Boolean = {
    val resultFile = resultFilename(datasetFilename)
    new File(resultFile).exists()
  }

  def evaluate(testFilename: String): MorphExperiment = {

    if (!exists(testFilename)) {
      val analyzer = train()
      val sentences = Dataset.readSentences(testFilename)
      sentences.par.map(sentence => measure(analyzer, sentence))
        .toArray
        .foreach(other => result.merge(other))
      save(testFilename)

    }
    this
  }

  def modelRanker(): MorphPredictor = {
    new MorphRanker(params)
      .setAmbiguityFreq(params.maxPairAmbiguity)
      .setOrder(params.wordSkipOrder)
      .setWindow(params.maxTokenWindow)
      .setIsTraining(true)
      .load()
  }


  def modelGMM(): MorphPredictor = {
    new GMMRanker(params)
      .setAmbiguityFreq(params.maxPairAmbiguity)
      .setIsTraining(true)
      .load()
  }

  def modelSelfAttention(): MorphPredictor = {
    new DeepSARanker(params)
      .setIsTraining(true)
      .load()
  }

  def modelElmo(): MorphPredictor = {
    new DeepELMORanker(params)
      .setIsTraining(true)
      .load()
  }

  def model(): MorphPredictor = {
    if ("gmm".equals(params.model)) modelGMM()
    else if ("self-attention".equals(params.model)) modelSelfAttention()
    else if ("elmo".equals(params.model)) modelElmo()
    else modelRanker()
  }

  def train(): MorphPredictor = {

    val morphRanker: MorphPredictor = model()


    val modelFilename = params.rankModelFilename()

    if (!new File(modelFilename).exists()) {
      println("Filtering sentences")
      val files = new File(params.sentenceTrainFolder).listFiles()
      files.foreach(file => {
        println("Crr file: " + file.getName)
        (0 until params.maxEpocs).par.map { i => {
          println("DAG Epoc: " + i)
          val index = i * params.maxSentences
          val crrSentences = Source.fromFile(file).getLines()
            .filter(line => line.length <= params.maxSentenceLength)
            .zipWithIndex.filter(_._2 >= index).take(params.maxSentences).map(_._1)
            .toArray

          System.gc()
          model().train(crrSentences)
        }
        }.toArray.foreach(other => morphRanker.merge(other))
      })

      morphRanker.trigger().save()


    }

    morphRanker

  }

  def incrementCount(voteMap: Map[Int, Array[Count]], count: Count, index: Int): Map[Int, Array[Count]] = {
    val array = voteMap.getOrElse(index, Array(count))
    val found = array.find(_.equals(count))
    if (found.isDefined) {
      found.get.inc()
    }

    voteMap.updated(index, array)
  }


  def measure(morphRanker: MorphPredictor, sentence: Sentence): Result = {

    val crrResult = Result(params)
    val analysis = sentence.words.map(word => word.toRankWord(params.maxSliceNgram))
    val actuals = sentence.words

    val voteMax = morphRanker.infer(analysis)


    var voteMap = Map[Int, Array[Count]]()
    var topMap = Map[Int, Array[String]]()
    var originalMap = Map[Int, Word]()
    var countAnalysis = 0

    voteMax.foreach { case (_, rankMorphemes) => {
      val crrMax = rankMorphemes.last
      val crrTopPredictions = rankMorphemes.reverse.take(5).map(_.analysis)
      val index = crrMax.tokenIndex
      val originalWord = actuals(index)
      val predictedString = crrMax.analysis
      val trueFound = crrResult.mostSimilar(originalWord, predictedString)
      val count = Count(trueFound, 0)
      voteMap = incrementCount(voteMap, count, index)
      originalMap = originalMap.updated(index, originalWord)
      topMap = topMap.updated(index, crrTopPredictions)
    }
    }

    val foundMap = voteMap.view.mapValues(_.sortBy(_.count).last).toMap

    var totalAmbiguous = 0
    var totalTrueAmbigous = 0
    var top5Count = 0
    originalMap.foreach { case (index, original) => {
      val predicted = foundMap.get(index).get
      val predictedTop5 = topMap.get(index).get
      val tcount = crrResult.addPrediction(original, predicted.item)
      val (tambiguous, ttrueambiguous) = crrResult.addAmbiguous(original, predicted.item)
      val ttop5 = crrResult.addTop5(original, predictedTop5)
      countAnalysis = countAnalysis + tcount._1
      totalAmbiguous = totalAmbiguous + tambiguous
      totalTrueAmbigous = totalTrueAmbigous + ttrueambiguous
      top5Count = top5Count + ttop5
    }
    }

    crrResult.incrementAmbiguous(totalAmbiguous, totalTrueAmbigous)
    crrResult.increment(countAnalysis == originalMap.size)
    crrResult.incrementTop5(top5Count)
    crrResult
  }


  def constructTraining(): this.type = {

    if (!new File(params.trainingFilename).exists()) {
      println("Building vocabulary")
      val vocabulary = tests().par.flatMap(testFilename => {
        Dataset.readSentences(testFilename).par
          .flatMap(sentence => sentence.words.map(_.text))
          .toSet
      }).toSet.toArray

      println("Vocabulary size: " + vocabulary.size)
      println("Building training corpus")

      val pw = new PrintWriter(params.trainingFilename)

      vocabulary.zipWithIndex.par.flatMap(crrPair => {
        println("Current word: " + crrPair._2)
        Source.fromFile(params.sentenceFilename).getLines()
          .filter(_.length < params.maxSentenceLength).filter(line => line.contains(crrPair._1)).take(params.maxWordSamples)
      }).toArray.foreach(line => pw.println(line))

      pw.close()

    }

    this
  }
}


object MorphExperiment {

  def tests(): Array[String] = {
    val params = new Params()
    params.testFilenames
  }

  def rankSA(): Array[Params] = {

    val maxWindows = new Params().maxWindowArray
    maxWindows.map(maxWindow => {

      val params = new Params()
      params.model = "self-attention"
      params.maxPairAmbiguity = 1
      params.maxTokenWindow = maxWindow
      params.maxSentenceAmbiguity = 1000000
      params.skipHeadAmbiguity = 1
      params.maxFeatures = 5000
      params.maxLabels = 1000
      params.maxSentences = 1000
      params.maxSentenceLength = 150
      params.maxEpocs = 40
      params.maxNeuralEpocs = 1
      params.maxSliceNgram = 2
      params.batchSize = 96
      params.hiddenSize = 200
      params.lrate = 0.01

      params.sentenceFilename = params.sentenceLabelFilename
      params

    })
  }

  def rankElmo(): Array[Params] = {

    val maxWindows = new Params().maxWindowArray
    maxWindows.map(maxWindow => {

      val params = new Params()
      params.model = "elmo"
      params.maxPairAmbiguity = 1
      params.maxTokenWindow = maxWindow
      params.maxSentenceAmbiguity = 1000000
      params.skipHeadAmbiguity = 1
      params.maxFeatures = 5000
      params.maxLabels = 1000
      params.maxSentences = 100
      params.maxSentenceLength = 150
      params.maxEpocs = 96
      params.maxNeuralEpocs = 1
      params.maxSliceNgram = 2
      params.batchSize = 48
      params.hiddenSize = 200
      params.lrate = 0.01

      params.sentenceFilename = params.sentenceLabelFilename
      params

    })
  }

  def rankParams(): Array[Params] = {
    val prunningRatios = new Params().prunningRatios
    val maxWindows = new Params().maxWindowArray
    val maxSlices = new Params().maxSliceArray
    val skipHeads = new Params().skipHeadArray

    skipHeads.flatMap(skipHead => {
      maxSlices.flatMap(maxSlice => {
        maxWindows.flatMap(maxWindow => {
          prunningRatios.map(prate => {

            val params = new Params()
            params.model = "rank"
            params.prunningRatio = prate
            params.maxPairAmbiguity = 1
            params.skipHeadAmbiguity = skipHead
            params.maxTokenWindow = maxWindow
            params.maxSentenceAmbiguity = Int.MaxValue
            params.maxSentences = 1000
            params.maxRankIters = 10
            params.maxEpocs = 200
            params.maxSliceNgram = maxSlice
            params.sentenceFilename = params.sentenceLabel4Filename
            params

          })
        })
      })

    })

  }


  def experimentRanking(): Unit = {

    val pars = rankParams().take(1)
    val testFilenames = tests()

    pars.par.foreach(param => {

      testFilenames.filter(testFilename => {
        !new MorphExperiment(param).exists(testFilename)
      }).map(testFilename => {
        new MorphExperiment(param)
          .constructTraining()
          .evaluate(testFilename)
          .saveAppend()
      })

    })


  }

  def experimentSelfAttention(): Unit = {

    val params = rankSA()
    val testFilenames = tests()
    params.foreach(param => {
      testFilenames.foreach(testFilename => {
        new MorphExperiment(param)
          .constructTraining()
          .evaluate(testFilename)
      })
    })

  }

  def experimentElmo(): Unit = {

    val params = rankElmo()
    val testFilenames = tests()
    params.foreach(param => {
      testFilenames.foreach(testFilename => {
        new MorphExperiment(param)
          .constructTraining()
          .evaluate(testFilename)
      })
    })

  }

  def main(args: Array[String]): Unit = {

    System.setProperty("org.bytedeco.openblas.load", "mkl")
    experimentRanking()
    //constructTraining()
    //experimentGMM()
    //experimentElmo()
  }
}