package morphology.ranking

import morphology.data.{Params, Sentence}
import org.deeplearning4j.nn.graph.ComputationGraph
import org.deeplearning4j.optimize.listeners.PerformanceListener
import org.deeplearning4j.util.ModelSerializer
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.indexing.NDArrayIndex

import java.io.File

abstract class DeepRanker(params:Params) extends MorphPredictor(params) {

  var dictionary = Map[String, Int]("dummy"-> 0)
  var labelDictionary = Map[String, Int]("dummy" -> 0)
  var labelSwap = Map[Int, String]()
  var samples = Array[Array[RankMorpheme]]()
  var computationGraph: ComputationGraph = null

  def indexLabel(indices: Array[Int], windowSize: Int): INDArray = {
    val ndarray = Nd4j.zeros(1, 1,  windowSize)
    indices.zipWithIndex.foreach(indicePair=>{
      ndarray.put(Array(NDArrayIndex.point(0), NDArrayIndex.point(0), NDArrayIndex.point(indicePair._2)), indicePair._1.toFloat)
    })
    ndarray
  }

  def bowFeatures(indices: Array[Int], windowSize:Int, maxSize: Int): INDArray = {

    val ndarray = Nd4j.zeros(1, maxSize, windowSize)
    indices.zipWithIndex.foreach{case(indice, windex) => {
      ndarray.put(Array(NDArrayIndex.point(0), NDArrayIndex.point(indice), NDArrayIndex.point(windex)), 1f)
    }}

    ndarray
  }

  def indexFeatures(indices: Array[Int], windowSize: Int): INDArray = {

    val ndarray = Nd4j.zeros(1, 1, windowSize)
    indices.zipWithIndex.foreach{case(indice, windex) => {
      ndarray.put(Array(NDArrayIndex.point(0), NDArrayIndex.point(0), NDArrayIndex.point(windex)), indice.toFloat)
    }}

    ndarray
  }

  def indexFeatures(indices: Array[Int], maxSize:Int, windowSize: Int): INDArray = {

    val ndarray = Nd4j.zeros(1, maxSize, windowSize)
    indices.zipWithIndex.foreach{case(indice, windex) => {
      ndarray.put(Array(NDArrayIndex.point(0), NDArrayIndex.point(indice), NDArrayIndex.point(windex)), 1f)
    }}

    ndarray
  }

  def maskInput(maxWindowSize:Int, actualSize:Int):INDArray={
    val maskArray = Nd4j.zeros(1, maxWindowSize)
    for(i<-0 until actualSize){
      maskArray.put(Array(NDArrayIndex.point(0), NDArrayIndex.point(i)), 1f)
    }
    maskArray
  }

  def maskLabel(maxWindowSize:Int, actualSize:Int):INDArray={
    val maskArray = Nd4j.zeros(1, maxWindowSize)
    for(i<-0 until actualSize){
      maskArray.put(Array(NDArrayIndex.point(0), NDArrayIndex.point(i)), 1f)
    }
    maskArray
  }


  def bowLabel(indices: Array[Int], maxSize: Int): INDArray = {
    val ndarray = Nd4j.zeros(1, maxSize)
    indices.foreach(indice=>{
      ndarray.put(Array(NDArrayIndex.point(0), NDArrayIndex.point(indice)), 1f)
    })

    ndarray
  }

  override def trigger(): MorphPredictor = {
    train()
    labelSwap = labelDictionary.map(_.swap)
    this
  }

  override def save(): this.type = {
    println("Saving model")
    ModelSerializer.writeModel(computationGraph, params.neuralModelFilename(), true)
    this
  }

  override def load(): this.type = {

    this

  }

  override def merge(other: MorphPredictor): MorphPredictor = {
    val otherPredictor = other.asInstanceOf[DeepRanker]
    samples = samples ++ otherPredictor.samples
    otherPredictor.labelDictionary.foreach{case(labelItem, index)=>{
      if(!labelDictionary.contains(labelItem)) labelDictionary = labelDictionary.updated(labelItem, labelDictionary.size)
    }}

    otherPredictor.dictionary.foreach{case(featureItem, index)=>{
      if(!dictionary.contains(featureItem)) dictionary = dictionary.updated(featureItem, dictionary.size)
    }}

    this
  }

  def train(): this.type = {

    var i = 0
    val fname = params.neuralModelFilename()
    val modelFile = new File(fname)
    println("LSTM filename: " + fname)
    if (!(modelFile.exists())) {

      computationGraph = model()
      computationGraph.addListeners(new PerformanceListener(2, true))

      val multiDataSetIterator = iterator()
      while (i < params.maxNeuralEpocs) {

        println("Epoc : " + i)
        computationGraph.fit(multiDataSetIterator)
        multiDataSetIterator.reset()
        i = i + 1
      }

      System.gc()
      //save()
    }
    else {
      computationGraph = ModelSerializer.restoreComputationGraph(modelFile)
    }
    this

  }

  def model(): ComputationGraph

  def iterator(): MultiDataSetIterator

}
