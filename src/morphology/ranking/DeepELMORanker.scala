package morphology.ranking

import morphology.data.{Params, Sentence}
import org.deeplearning4j.nn.conf.{NeuralNetConfiguration, RNNFormat}
import org.deeplearning4j.nn.conf.graph.MergeVertex
import org.deeplearning4j.nn.conf.graph.rnn.LastTimeStepVertex
import org.deeplearning4j.nn.conf.inputs.InputType
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer.AlgoMode
import org.deeplearning4j.nn.conf.layers.{EmbeddingSequenceLayer, LSTM, OutputLayer, RnnOutputLayer}
import org.deeplearning4j.nn.graph.ComputationGraph
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.MultiDataSet
import org.nd4j.linalg.dataset.api.MultiDataSetPreProcessor
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.indexing.NDArrayIndex
import org.nd4j.linalg.learning.config.Adam
import org.nd4j.linalg.lossfunctions.LossFunctions

import java.io.{ObjectInputStream, ObjectOutputStream}

class DeepELMORanker(params: Params) extends DeepRanker(params) {


  override def construct(sequence: Sentence): MorphPredictor = {
    val rankWords = sequence.words.map(word => word.toRankWord(params.maxSliceNgram))
    rankWords.sliding(params.maxTokenWindow, 1).foreach(windowWords => {

      analyzer.combinatoricMorpheme(windowWords).foreach(rankSequence => {
        samples = samples :+ rankSequence.rankMorphemes

        val (forwardFeatures, labels) = processForward(rankSequence.rankMorphemes)
        forwardFeatures.foreach(item => {
          if (!dictionary.contains(item)) {
            dictionary = dictionary.updated(item, dictionary.size)
          }
        })

        labels.foreach(item => {
          if (!labelDictionary.contains(item)) {
            labelDictionary = labelDictionary.updated(item, labelDictionary.size)
          }
        })

      })
    })


    this

  }


  override def save(objectStream: ObjectOutputStream): MorphPredictor = this

  override def load(objectStream: ObjectInputStream): MorphPredictor = this

  def processForward(sentence: Array[RankMorpheme]): (Array[String], Array[String]) = {
    val featureSequence = sentence.take(sentence.length - 1).flatMap(rankMorpheme => rankMorpheme.labels.filter(!_.lemmaLabel)).map(i=> ">>" + i.tags)
    val targetSequence = sentence.last.labels.filter(!_.lemmaLabel).map(_.tags)
    (featureSequence, targetSequence)
  }

  def processBackward(sentence: Array[RankMorpheme]): (Array[String], Array[String]) = {
    val reverseSentence = sentence.reverse
    val featureSequence = reverseSentence.take(sentence.length - 1).flatMap(rankMorpheme => rankMorpheme.labels.reverse.filter(!_.lemmaLabel)).map(i=> ">>" + i.tags)
    val targetSequence = reverseSentence.head.labels.filter(!_.lemmaLabel).map(_.tags)
    (featureSequence, targetSequence)
  }

  def obtainScore(voteMap:Map[Int, Array[RankMorpheme]], inputMap:Map[Int, (INDArray, INDArray)]):Unit={

    inputMap.foreach {case(index, (forwardArray, backwardArray))=> {
      val outputArray = computationGraph.output(false, forwardArray, backwardArray)
        .head

     val outputVector = outputArray.toFloatVector


      val outputTags = outputVector.zipWithIndex.filter(_._1 >= 0.5)
        .map(pair=>labelSwap(pair._2))

      voteMap(index).foreach(targetMorpheme=>{
        val rankScore = targetMorpheme.rankBy(outputTags)
        targetMorpheme.incRank(rankScore)
      })
    }}
  }

  def inputForward(sentence: Array[RankMorpheme]): Map[Int, INDArray] = {

    sentence.zipWithIndex.sliding(params.maxTokenWindow, 1)
      .map(rankMorphemeArray => {
        val rankMorpheme = rankMorphemeArray.map(_._1)
        val predictionIndex = rankMorphemeArray.last._2
        val (features, _) = processForward(rankMorpheme)
        val featureIndices = features.map(item=> dictionary.getOrElse(item, 0)).take(params.maxLabelWindow)
        val featureArray = indexFeatures(featureIndices, params.maxLabelWindow)

        (predictionIndex, featureArray)
      }).toMap
  }

  def inputBackward(sentence: Array[RankMorpheme]): Map[Int, INDArray] = {
    sentence.zipWithIndex.sliding(params.maxTokenWindow, 1)
      .map(rankMorphemeArray => {
        val rankMorpheme = rankMorphemeArray.map(_._1)
        val predictionIndex = rankMorphemeArray.last._2
        val (features, _) = processBackward(rankMorpheme)
        val featureIndices = features.map(item=> dictionary.getOrElse(item, 0)).take(params.maxLabelWindow)
        val featureArray = indexFeatures(featureIndices, params.maxLabelWindow)

        (predictionIndex, featureArray)
      }).toMap
  }

  override def infer(results: Array[RankWord]): Map[Int, Array[RankMorpheme]] = {
    val voteMap = results.zipWithIndex.map(rankWordPair => rankWordPair._2 -> rankWordPair._1.rankMorphemes).toMap

    results.sliding(params.maxTokenWindow, 1).foreach(sequence=>{
      val combinations = analyzer.combinatoricMorpheme(sequence)
      combinations.foreach(rankSequence=>{
        val forwardMap = inputForward(rankSequence.rankMorphemes)
        val backwardMap = inputBackward(rankSequence.rankMorphemes)
        val mergeMap = forwardMap.map{case(index, forwardArray)=> (index -> (forwardArray, backwardMap(index)))}
        obtainScore(voteMap, mergeMap)
      })
    })

    voteMap.view.mapValues(rankMorphemes=> rankMorphemes.sorted)
      .toMap
  }

  def vectorizeForward(sentence: Array[RankMorpheme]): Array[(INDArray, INDArray, INDArray, INDArray)] = {
    sentence.sliding(params.maxTokenWindow, 1)
      .map(rankMorpheme => {
        val (features, labels) = processForward(rankMorpheme)
        val featureIndices = features.map(item=> dictionary.getOrElse(item, 0)).take(params.maxLabelWindow)
        val labelIndices = labels.map(item=> labelDictionary.getOrElse(item, 0))
        val featureArray = indexFeatures(featureIndices, params.maxLabelWindow)
        val labelArray = bowLabel(labelIndices,params.maxLabels)
        val maskInputArray = maskInput(params.maxLabelWindow, featureIndices.length)
        val maskLabelArray = maskInput(1, 1)

        (featureArray, labelArray, maskInputArray, maskLabelArray)
      })
      .toArray
  }

  def vectorizeBackward(sentence: Array[RankMorpheme]): Array[(INDArray, INDArray, INDArray, INDArray)] = {
    sentence.sliding(params.maxTokenWindow, 1)
      .map(rankMorpheme => {
        val (features, labels) = processBackward(rankMorpheme)
        val featureIndices = features.map(item => dictionary.getOrElse(item, 0)).take(params.maxLabelWindow)
        val labelIndices = labels.map(item => labelDictionary.getOrElse(item, 0))

        val featureArray = indexFeatures(featureIndices, params.maxLabelWindow)
        val labelArray = bowLabel(labelIndices,params.maxLabels)
        val maskArray = maskInput(params.maxLabelWindow, featureIndices.length)
        val maskLabelArray = maskInput(1,1)
        (featureArray, labelArray, maskArray, maskLabelArray)
      })
      .toArray
  }

  override def model(): ComputationGraph = {
    val conf = new NeuralNetConfiguration.Builder()
      .cudnnAlgoMode(AlgoMode.NO_WORKSPACE)
      .updater(new Adam.Builder().learningRate(params.lrate).build())
      .dropOut(0.5)
      .graphBuilder()
      .allowDisconnected(true)
      .addInputs("left", "right")
      .addVertex("stack", new org.deeplearning4j.nn.conf.graph.StackVertex(), "left", "right")
      .addLayer("embedding", new EmbeddingSequenceLayer.Builder().inputLength(params.maxLabelWindow)
        .nIn(params.maxFeatures).nOut(params.hiddenSize).build(),
        "stack")
      .addVertex("leftemb", new org.deeplearning4j.nn.conf.graph.UnstackVertex(0, 2), "embedding")
      .addVertex("rightemb", new org.deeplearning4j.nn.conf.graph.UnstackVertex(0, 2), "embedding")
      //can use any label for this
      .addLayer("leftout", new LSTM.Builder().nIn(params.hiddenSize).nOut(params.hiddenSize)
        .activation(Activation.RELU)
        .build(), "leftemb")
      .addLayer("rightout", new LSTM.Builder().nIn(params.hiddenSize).nOut(params.hiddenSize)
        .activation(Activation.RELU)
        .build(), "rightemb")
      .addVertex("merge", new MergeVertex(), "leftout", "rightout")
      .addLayer("output-lstm", new LSTM.Builder().nIn(params.hiddenSize).nOut(params.hiddenSize)
        .activation(Activation.RELU)
        .build(), "merge")
      .addVertex("output-last", new LastTimeStepVertex("left"), "output-lstm")
      .addLayer("output",
        new OutputLayer.Builder(LossFunctions.LossFunction.XENT)
          .activation(Activation.SIGMOID)
          .nOut(params.maxLabels)
          .build(), "output-last")
      .setOutputs("output")
      .setInputTypes(InputType.recurrent(params.maxFeatures),
        InputType.recurrent(params.maxFeatures))
      .build()

    new ComputationGraph(conf)
  }

  override def iterator(): MultiDataSetIterator = {

    new MultiDataSetIterator {

      var lines = samples.iterator

      override def next(i: Int): MultiDataSet = {
        var cnt = 0
        var trainForwardStack = Array[INDArray]()
        var trainBackwardStack = Array[INDArray]()
        var trainOutputStack = Array[INDArray]()
        var maskForwardStack = Array[INDArray]()
        var maskBackwardStack = Array[INDArray]()
        var maskLabelStack = Array[INDArray]()

        while (cnt < params.batchSize && hasNext) {

          val sentenceMorphemes = lines.next()
          val vectorsForward = vectorizeForward(sentenceMorphemes)
          val vectorsBackward = vectorizeBackward(sentenceMorphemes)

          vectorsForward.foreach{case(input, output, maskInput, maskLabel)=>{
            trainForwardStack = trainForwardStack :+ input
            trainOutputStack = trainOutputStack :+ output
            maskForwardStack = maskForwardStack :+ maskInput
            maskLabelStack = maskLabelStack :+ maskLabel
          }}

          vectorsBackward.foreach{case(input, _, maskInput, _)=>{
            trainBackwardStack = trainBackwardStack :+ input
            maskBackwardStack = maskBackwardStack :+ maskInput
          }}

          cnt += 1
        }

        val trainForwardVector = Nd4j.vstack(trainForwardStack: _*)
        val trainBackwardVector = Nd4j.vstack(trainBackwardStack: _*)
        val trainMaskForwardVector = Nd4j.vstack(maskForwardStack: _*)
        val trainMaskBackwardVector = Nd4j.vstack(maskBackwardStack: _*)
        val trainOutputVector = Nd4j.vstack(trainOutputStack: _*)
        val trainMaskLabelVector = Nd4j.vstack(maskLabelStack: _*)

        val inputArray = Array(trainForwardVector, trainBackwardVector)
        val inputMaskArray = Array(trainMaskForwardVector, trainMaskBackwardVector)
        val labelArray = Array(trainOutputVector)
        val labelMaskArray = Array(trainMaskLabelVector)

        new org.nd4j.linalg.dataset.MultiDataSet(inputArray, labelArray, inputMaskArray, labelMaskArray)
      }

      override def setPreProcessor(multiDataSetPreProcessor: MultiDataSetPreProcessor): Unit = {}

      override def getPreProcessor: MultiDataSetPreProcessor = null

      override def resetSupported(): Boolean = true

      override def asyncSupported(): Boolean = false

      override def reset(): Unit = {
        lines = samples.iterator
      }

      override def hasNext: Boolean = lines.hasNext

      override def next(): MultiDataSet = next(0)
    }
  }
}
