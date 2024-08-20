package morphology.ranking

import morphology.data.{Label, Params, Sentence}
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.inputs.InputType
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer.AlgoMode
import org.deeplearning4j.nn.conf.layers.{DenseLayer, EmbeddingSequenceLayer, GlobalPoolingLayer, LSTM, OutputLayer, PoolingType, SelfAttentionLayer}
import org.deeplearning4j.nn.graph.ComputationGraph
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.listeners.PerformanceListener
import org.deeplearning4j.util.ModelSerializer
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.api.buffer.DataType
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.MultiDataSet
import org.nd4j.linalg.dataset.api.MultiDataSetPreProcessor
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.indexing.{INDArrayIndex, NDArrayIndex}
import org.nd4j.linalg.learning.config.Adam
import org.nd4j.linalg.lossfunctions.LossFunctions

import java.io.{File, FileInputStream, FileOutputStream, ObjectInputStream, ObjectOutputStream}
import scala.io.Source

class DeepSARanker(params: Params) extends DeepRanker(params) {


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

        val (backwardFeatures, _) = processBackward(rankSequence.rankMorphemes)
        backwardFeatures.foreach(item => {
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


  def processForward(sentence: Array[RankMorpheme]): (Array[String], Array[String]) = {
    val featureSequence = sentence.take(sentence.length - 1).flatMap(rankMorpheme => rankMorpheme.labels.filter(!_.lemmaLabel)).map(i=> ">>" + i.tags)
    val targetSequence = sentence.last.labels.filter(!_.lemmaLabel).map(_.tags)
    (featureSequence, targetSequence)
  }

  def processBackward(sentence: Array[RankMorpheme]): (Array[String], Array[String]) = {
    val reverseSentence = sentence.reverse
    val featureSequence = reverseSentence.take(sentence.length - 1).flatMap(rankMorpheme => rankMorpheme.labels.reverse.filter(!_.lemmaLabel)).map(i=> "<<" + i.tags)
    val targetSequence = reverseSentence.last.labels.filter(!_.lemmaLabel).map(_.tags)

    (featureSequence, targetSequence)
  }


  override def save(objectStream: ObjectOutputStream): MorphPredictor = this

  override def load(objectStream: ObjectInputStream): MorphPredictor = this

  override def infer(results: Array[RankWord]): Map[Int, Array[RankMorpheme]] = {
    val voteMap = results.zipWithIndex.map(rankWordPair => rankWordPair._2 -> rankWordPair._1.rankMorphemes).toMap

    results.sliding(params.maxTokenWindow, 1).foreach(sequence=>{
      val combinations = analyzer.combinatoricMorpheme(sequence)
      combinations.foreach(rankSequence=>{
        val forwardMap = inputForward(rankSequence.rankMorphemes)
        val backwardMap = inputBackward(rankSequence.rankMorphemes)
        obtainScore(voteMap, forwardMap)
        obtainScore(voteMap, backwardMap)
      })
    })

    voteMap.view.mapValues(rankMorphemes=> rankMorphemes.sorted)
      .toMap
  }

  def obtainScore(voteMap:Map[Int, Array[RankMorpheme]], inputMap:Map[Int, INDArray]):Unit={

    inputMap.foreach {case(index, array)=> {
      val outputArray = computationGraph.output(false, array)
        .head
        .toFloatVector
      val outputTags = outputArray.zipWithIndex.filter(_._1 >= 0.5)
        .map(pair=>labelSwap(pair._2))

      voteMap(index).foreach(targetMorpheme=>{
        val rankScore = targetMorpheme.rankBy(outputTags)
        targetMorpheme.incRank(rankScore)
      })
    }}
  }


  def vectorizeForward(sentence: Array[RankMorpheme]): Array[(INDArray, INDArray, INDArray, INDArray)] = {
    sentence.sliding(params.maxTokenWindow, 1)
      .map(rankMorpheme => {
        val (features, labels) = processForward(rankMorpheme)
        val featureIndices = features.map(item=> dictionary.getOrElse(item, 0)).take(params.maxLabelWindow)
        val labelIndices = labels.map(item=> labelDictionary.getOrElse(item, 0))
        val featureArray = indexFeatures(featureIndices, params.maxLabelWindow)
        val labelArray = bowLabel(labelIndices, params.maxLabels)
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
        val featureIndices = features.map(item=> dictionary.getOrElse(item, 0)).take(params.maxLabelWindow)
        val labelIndices = labels.map(item=> labelDictionary.getOrElse(item, 0))


        val featureArray = indexFeatures(featureIndices, params.maxLabelWindow)
        val labelArray = bowLabel(labelIndices, params.maxLabels)
        val maskArray = maskInput(params.maxLabelWindow, featureIndices.length)
        val maskLabelArray = maskInput(1, 1)
        (featureArray, labelArray, maskArray, maskLabelArray)
      })
      .toArray
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
        val predictionIndex = rankMorphemeArray.head._2
        val (features, _) = processBackward(rankMorpheme)
        val featureIndices = features.map(item=> dictionary.getOrElse(item, 0)).take(params.maxLabelWindow)
        val featureArray = indexFeatures(featureIndices, params.maxLabelWindow)

        (predictionIndex, featureArray)
      }).toMap
  }

  def iterator(): MultiDataSetIterator = {

    new MultiDataSetIterator {

      var lines = samples.iterator

      override def next(i: Int): MultiDataSet = {
        var cnt = 0
        var trainStack = Array[INDArray]()
        var trainOutputStack = Array[INDArray]()
        var maskInputStack = Array[INDArray]()
        var maskLabelStack = Array[INDArray]()

        while (cnt < params.batchSize && hasNext) {

          val sentenceMorphemes = lines.next()
          val vectorsForward = vectorizeForward(sentenceMorphemes)
          val vectorsBackward = vectorizeBackward(sentenceMorphemes)

          val vectors = vectorsForward ++ vectorsBackward
          vectors.foreach{case(input, output, maskInput, maskLabel)=>{
            trainStack = trainStack :+ input
            trainOutputStack = trainOutputStack :+ output
            maskInputStack = maskInputStack :+ maskInput
            maskLabelStack = maskLabelStack :+ maskLabel
          }}

          cnt += 1
        }

        val trainVector = Nd4j.vstack(trainStack: _*)
        val trainOutputVector = Nd4j.vstack(trainOutputStack: _*)
        val trainMaskInputVector = Nd4j.vstack(maskInputStack: _*)
        val trainMaskLabelVector = Nd4j.vstack(maskLabelStack: _*)
        new org.nd4j.linalg.dataset.MultiDataSet(trainVector, trainOutputVector, trainMaskInputVector, trainMaskLabelVector)
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



  def model(): ComputationGraph = {

    val conf = new NeuralNetConfiguration.Builder()
      .cudnnAlgoMode(AlgoMode.PREFER_FASTEST)
      .dataType(DataType.FLOAT)
      .activation(Activation.TANH)
      .updater(new Adam(params.lrate))
      .weightInit(WeightInit.XAVIER)
      .graphBuilder()
      .addInputs("input")
      .setOutputs("output")
      .layer("embedding", new EmbeddingSequenceLayer.Builder()
        .inputLength(params.maxLabelWindow)
        .nIn(params.maxFeatures)
        .nOut(params.hiddenSize).build(), "input")
      .layer("input-lstm", new LSTM.Builder().nIn(params.hiddenSize).nOut(params.hiddenSize)
        .activation(Activation.TANH).build, "embedding")
      .layer("attention", new SelfAttentionLayer.Builder().nOut(params.hiddenSize).nHeads(params.nHeads).projectInput(true).build(), "input-lstm")
      .layer("pooling", new GlobalPoolingLayer.Builder().poolingType(PoolingType.MAX).build(), "attention")
      .layer("dense_base", new DenseLayer.Builder().nIn(params.hiddenSize).nOut(params.hiddenSize).activation(Activation.RELU).build(), "pooling")
      .layer("dense", new DenseLayer.Builder().nIn(params.hiddenSize).nOut(params.hiddenSize).activation(Activation.RELU).build(), "dense_base")
      .layer("output", new OutputLayer.Builder().nIn(params.hiddenSize).nOut(params.maxLabels).activation(Activation.SIGMOID)
        .lossFunction(LossFunctions.LossFunction.XENT).build, "dense")
      .setInputTypes(InputType.recurrent(params.maxFeatures))
      .build()

    new ComputationGraph(conf)
  }





}
