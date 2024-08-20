package morphology.data

import morphology.ranking.{DataRank, Vertex}
import zemberek.morphology.TurkishMorphology

import java.io.{ObjectInputStream, ObjectOutputStream}
import scala.reflect.ClassTag

class MorphRanker {

  private var default: String = "none"

  var map = Map[String, Vertex]()
  var order: Array[Int] = Array[Int](1, 2, 3)

  var window: Int = 20
  var dependencyCategory: (Int) => String = linkMarker

  var triggerFreq = 100
  var triggerSamples: Array[Array[String]] = Array()
  var processCount = 0

  var doTrain = true

  def setLinkMarking(fun: (Int) => String): this.type = {
    this.dependencyCategory = fun
    this
  }

  def setOrder(order: Array[Int]): this.type = {
    this.order = order
    this
  }

  def setWindow(window: Int): this.type = {
    this.window = window
    this
  }

  def setTriggerFreq(freq: Int): this.type = {
    triggerFreq = freq
    this
  }

  def setDoTrain(value: Boolean): this.type = {
    doTrain = value
    this
  }

  def getLabel(): String = "morph"

  def update(name: String, value: String): Vertex = {

    if (map.contains(value)) {
      map(value)
    }
    else {
      val vertex = Vertex(name, value)
      map = map + (value -> vertex)
      vertex
    }
  }

  def update(vertex: Vertex): Vertex = {
    val value = vertex.value
    if (map.contains(value)) {
      map(value)
    }
    else {
      map = map + (value -> vertex)
      vertex
    }
  }

  def getCopy(label: String, value: String): Vertex = {
    map.getOrElse(value, Vertex(label, value)).copy()
  }

  def rank(analysisSequence: Array[Array[String]]): Array[DataRank] = {

    if (doTrain) {
      construct(analysisSequence, order)
    }

    var rankArray = Array[DataRank]()
    analysisSequence.foreach { array => {
      val dscore = score(array, order)
      val darray = new DataRank(array, dscore)
      rankArray = rankArray :+ darray
      triggerSamples = triggerSamples :+ array
    }}

    trigger()
    rankArray
  }

  def save(objectStream: ObjectOutputStream): this.type = {

    normalize()

    objectStream.writeInt(map.size)
    map.foreach { case (key, vertex) => {
      objectStream.writeObject(key)
      vertex.save(objectStream)
    }
    }

    this
  }

  def load(objectStream: ObjectInputStream): this.type = {
    this.map = Map[String, Vertex]()
    val size = objectStream.readInt()

    for (i <- 0 until size) {
      val key = objectStream.readObject().asInstanceOf[String]
      val vertex = Vertex("none", null).load(objectStream)
      map = map + (key -> vertex)
    }

    this
  }

  def linkMarker(i: Int): String = {
    (if (i > 20) "Distant" else if (i > 15) "Non-Local" else "Local")
  }

  protected def constructOrder(linkName: String, sequence: Array[Array[String]], order: Int = 1): MorphRanker = {
    sequence.foreach(array => array.sliding(1 + order, 1).foreach(elem => {
      val a = elem.head
      val b = elem.last
      val avertex = update(getLabel(), a)
      val bvertex = update(getLabel(), b)
      avertex.add(linkName, bvertex)
    }))

    this
  }

  def construct(sequence: Array[Array[String]], order: Array[Int]): MorphRanker = {
    order.foreach(i => {
      constructOrder(linkMarker(i), sequence, i)
    })
    this
  }

  def trigger(): MorphRanker = {

    processCount = processCount + 1
    if (processCount >= triggerFreq) {
      println("Triggering normalization and dropout")
      normalize()
      triggerSamples.foreach(item => droupout(item))
      triggerSamples = Array()
      processCount = 0
    }
    this
  }

  def normalize(): MorphRanker = {
    map.foreach { case (b, vertex) => vertex.normalize() }
    this
  }

  def droupout(sequence: Array[String]): MorphRanker = {
    val mapVertex = sequence.map(item => item -> getCopy(getLabel(), item)).toMap
    val maxWindow = order.max

    val vertices = sequence.sliding(maxWindow, 1).map { subsequence => {
      (subsequence, score(subsequence, order))
    }
    }

    val removes = vertices.toArray.sortBy(pair => pair._2)
      .take(vertices.length / 2)
      .flatMap {
        case (sequence, rank) => {
          order.flatMap(i => sequence.sliding(i + 1, 1).map(elems => {
            val va = mapVertex(elems.head)
            val vb = mapVertex(elems.last)
            va.remove(vb, dependencyCategory(i))
          }))
        }
      }

    removes.foreach(vertex => update(vertex))
    this
  }

  protected def scoreByOrder(sequence: Array[String], iters: Int = 5, order: Int = 1): Double = {
    val mapVertex = sequence.map(item => item -> getCopy(getLabel(), item)).toMap

    for (i <- 0 until iters) {
      sequence.sliding(order + 1, 1).map(elems => {
        val va = mapVertex(elems.head)
        val vb = mapVertex(elems.last)
        val score = va.positive(vb, dependencyCategory(order))
        val rank = 0.15 * vb.rank + 0.85 * score * va.rank
        vb.setRank(rank)
      })
    }

    mapVertex.map(_._2.rank).sum
  }

  def score(sequence: Array[String], order: Array[Int]): Double = {
    order.map(i => scoreByOrder(sequence, 10, i)).last
  }

  def scoreMax(linkName: String, sequence: Array[Array[String]], order: Array[Int]): Array[String] = {
    sequence.map(items => (items, score(items, order))).sortBy(_._2)
      .last._1
  }
}


object DataRankProcessor {
  def main(args: Array[String]): Unit = {
    val order = Array(0, 1, 2, 3)
    val combinator = new DataCombinator()
    val analyzer = new DataMorphAnalyzer()
    val splitter = new DataStringFlatten().setPrepend(">>").setSplitBy("[\\+\\|]").setAppend("")
    val tokens = Array("Dün", "akşam", "çok", "zaman", "alan", "bir", "yoldaydım")
    val morphs = tokens.map(analyzer.analyze(TurkishMorphology.createWithDefaults(), _))
    val combs = combinator.combinations(morphs).map(sequence => splitter.process(sequence))
    val ranking = new DataStringRankProcessor().construct("stem", combs, order)
    combs.map(items => (items, ranking.score("stem", items, Array[Int](0)))).sortBy(_._2)
      .reverse.zipWithIndex
      .foreach { case ((item, value), index) => {
        println("Index: " + index + " Data: " + splitter.processReverse(item).mkString(" ") + ", Score:" + value)
      }
      }
  }
}
