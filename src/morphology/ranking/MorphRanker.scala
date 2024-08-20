package morphology.data

import morphology.morph.{MorphAnalyzer, Tokenizer}
import morphology.ranking.{DataRank, MorphPredictor, RankMorpheme, RankSequence, RankWord, Vertex, VertexKey}
import zemberek.morphology.TurkishMorphology

import java.io.{File, FileInputStream, FileOutputStream, ObjectInputStream, ObjectOutputStream}
import scala.collection.parallel.CollectionConverters.{ArrayIsParallelizable, IterableIsParallelizable}
import scala.io.Source
import scala.reflect.ClassTag
import scala.util.Random

class MorphRanker(params: Params) extends MorphPredictor(params) {

  var mapPositive = Map[String, Vertex]()
  var mapNegative = Map[String, Vertex]()
  var probMap = Map[Label, Double]()
  var count = 0
  var dependencyCategory: (Int) => String = linkMarker

  def update(name: String, label: Label): Vertex = {
    val value = label.tags
    if (mapPositive.contains(value)) {
      mapPositive(value)
    }
    else {
      val vertex = Vertex(name, value)
        .setLemma(label.lemmaLabel)
      mapPositive = mapPositive + (value -> vertex)
      vertex
    }
  }

  def update(vertex: Vertex): Vertex = {
    val value = vertex.value
    if (mapPositive.contains(value)) {
      mapPositive(value)
    }
    else {
      mapPositive = mapPositive + (value -> vertex)
      vertex
    }
  }
  /*
  def getCopy(value: String, size: Int): Vertex = {
    map.getOrElse(value, Vertex("morph", value).setRank(1d / size)).copy()
  }
  */


  def save(objectStream: ObjectOutputStream): this.type = {

    normalize()

    objectStream.writeInt(count)
    objectStream.writeInt(mapPositive.size)
    mapPositive.foreach { case (key, vertex) => {
      objectStream.writeObject(key)
      vertex.save(objectStream)
    }
    }

    objectStream.writeInt(mapNegative.size)
    mapNegative.foreach { case (key, vertex) => {
      objectStream.writeObject(key)
      vertex.save(objectStream)
    }
    }

    objectStream.writeInt(probMap.size)
    probMap.foreach { case (key, score) => {
      objectStream.writeObject(key)
      objectStream.writeDouble(score)
    }
    }
    this
  }

  def load(objectStream: ObjectInputStream): this.type = {
    this.mapPositive = Map[String, Vertex]()
    this.count = objectStream.readInt()

    val size = objectStream.readInt()
    for (i <- 0 until size) {
      val key = objectStream.readObject().asInstanceOf[String]
      val vertex = Vertex("none", null).load(objectStream)
      mapPositive = mapPositive + (key -> vertex)
    }

    val size2 = objectStream.readInt()
    for (i <- 0 until size2) {
      val key = objectStream.readObject().asInstanceOf[String]
      val vertex = Vertex("none", null).setLemma(false).load(objectStream)
      mapNegative = mapNegative + (key -> vertex)
    }

    val size3 = objectStream.readInt()
    for (i <- 0 until size3) {
      val key = objectStream.readObject().asInstanceOf[Label]
      val score = objectStream.readDouble()
      probMap = probMap + (key -> score)
    }

    this
  }

  def toStats(): DataStats = {

    val totalEdges = mapPositive.map(_._2.edges.size).sum
    val totalByEdgeType = mapPositive.flatMap(_._2.edges.map(_._1.label))
      .groupBy(item => item).view.mapValues(_.size)
      .toMap

    stats.totalTagDepencies = totalEdges
    stats.totalDistantLink = totalByEdgeType("Distant")
    stats.totalLocalLink = totalByEdgeType("Local")
    stats.totalNonLocalLink = totalByEdgeType("Neighbour")
    stats
  }

  def linkMarker(i: Int): String = {
    params.linkMarker(i)
  }


  override def construct(sentence: Sentence): MorphRanker.this.type = {

    sentence.toNonStemLabels(params.maxSliceNgram).foreach(label => {
      if (probMap.contains(label)) probMap = probMap.updated(label, probMap(label) + 1d)
      else probMap = probMap.updated(label, 1d)

      count += 1
    })

    sentence.toPairs(params.maxPairAmbiguity, params.maxSliceNgram, linkMarker, params.wordSkipOrder).foreach {
      case (linkName, aTag, bTag) => {
        val aVertex = update("morph", aTag)
        val bVertex = update("morph", bTag)
        aVertex.add(linkName, bVertex, 1.0)
      }
    }

    if (params.forwardBackward) {
      sentence.toReversePairs(params.maxPairAmbiguity, params.maxSliceNgram, linkMarker, params.wordSkipOrder).foreach {
        case (linkName, aTag, bTag) => {
          val aVertex = update("morph", aTag)
          val bVertex = update("morph", bTag)
          aVertex.add(linkName, bVertex, 1.0)
        }
      }
    }

    this
  }

  def score(rankSequence: RankSequence): RankSequence = {
    val forwardPath = rankSequence.copy()

    if (params.forwardBackward) {
      score(forwardPath.reverse(), params.wordSkipOrder, "Backward-")
      score(forwardPath.reverse(), params.wordSkipOrder, "Forward-")
      forwardPath
    }
    else {
      score(forwardPath, params.wordSkipOrder, "Forward-")
    }
  }

  def probScore(rankMorpheme: RankMorpheme): RankMorpheme = {

    val scores = rankMorpheme.labels.filter(!_.lemmaLabel).map(label => probMap.getOrElse(label, 1d) / count)
    val logSum = scores.map(i => -1 * math.log(i)).sum / scores.length
    rankMorpheme.setLoglikelihood(logSum)
    rankMorpheme
  }

  def infer(results: Array[RankWord]): Map[Int, Array[RankMorpheme]] = {

    var voteMap = (0 until results.length).map(i => (i -> Seq[RankMorpheme]())).toMap
    results.sliding(params.maxTokenWindow, 1).foreach { tokenWindow => {
      val windowCombinations = analyzer.combinatoricMorpheme(tokenWindow)
      windowCombinations.foreach(morphemePath => {
        val forwardPath = score(morphemePath)
        forwardPath.rankMorphemes.foreach(rankMorpheme => {
          var rankMorphemes = voteMap(rankMorpheme.tokenIndex)
          if (rankMorphemes.contains(rankMorpheme)) {
            val crrMorpheme = rankMorphemes.find(crrMorpheme => crrMorpheme.equals(rankMorpheme))
              .get
            crrMorpheme.addRank(rankMorpheme.rank)
          }
          else {
            rankMorphemes :+= rankMorpheme
            voteMap = voteMap.updated(rankMorpheme.tokenIndex, rankMorphemes)
          }
        })
      })
    }}


    val shuffleMap = voteMap.view.mapValues(s => {
      Random.shuffle(s).map(rankMorpheme => probScore(rankMorpheme))
    }).toMap

    val voteMax = shuffleMap.view.mapValues(pair => pair.sorted.toArray).toMap
    voteMax
  }


  override def trigger(): MorphRanker.this.type = {

    println("Triggering normalization and dropout")

    normalize()
    droupout()
    this
  }

  def buildNegative(): MorphRanker = {
    println("Building negatives")
    val keys = mapPositive.keys.map(mapPositive)
      .filter(v=> !v.isLemma)
      .map(_.value)
    mapNegative = keys.zipWithIndex.par.map(pair => {
      val srcString = pair._1
      println("Filtering: "+pair._2+"/"+keys.size)
      val negativeKeys = keys.par.filter(dstString => !srcString.equals(dstString))
        .filter(dstString => !mapPositive(srcString).edges.exists(pair=> pair._1.value.equals(dstString)))

      val negativeVertices = negativeKeys.map(negativeStr => VertexKey("Negative", negativeStr) -> 1.0).toArray.toMap
      val crrVertex = Vertex("morph", srcString).setEdges(negativeVertices)
      srcString -> crrVertex
    }).toArray.toMap

    this
  }

  def normalize(): MorphRanker = {
    println("Normalizing model...")
    mapPositive.foreach { case (b, vertex) => vertex.normalize() }
    this
  }

  override def merge(other: MorphPredictor): MorphRanker.this.type = {
    println("Merging ranker....")
    val otherRanker = other.asInstanceOf[MorphRanker]
    otherRanker.mapPositive.foreach { case (item, vertex) => {

      if (mapPositive.contains(item)) {
        mapPositive(item).merge(vertex)
      }
      else {
        mapPositive = mapPositive.updated(item, vertex)
      }
    }
    }

    otherRanker.probMap.foreach { case (item, otherCount) => {
      if (probMap.contains(item)) {
        probMap = probMap.updated(item, probMap(item) + otherCount)
      }
      else {
        probMap = probMap.updated(item, otherCount)
      }
    }
    }

    this.count += otherRanker.count
    this
  }

  protected def droupout(): MorphRanker = {
    println("Global prunning started")
    mapPositive.foreach(crrPair => {
      val crrVertex = crrPair._2
      val crrVertices = crrVertex.edges.toArray
      val keepSize = Math.max((crrVertices.length * params.prunningRatio).toInt, params.topEdges)
      val keep = crrVertices.sortBy(pair => pair._2).reverse
        .take(keepSize).toMap
      crrVertex.setEdges(keep)
    })

    this
  }


  def pairize(headMorpheme: RankMorpheme, lastMorpheme: RankMorpheme, linkName: String): Array[(String, String)] = {
    val pairs = if (linkName.contains("Local")) {
      headMorpheme.labels.map(_.tags).sliding(2, 1).map(seq => {
        (seq.head, seq.last)
      }).toArray
    }
    else {
      val heTags = headMorpheme.labels.map(_.tags)
      val laTags = lastMorpheme.labels.map(_.tags)
      heTags.flatMap(heTag => laTags.map(laTag => (heTag, laTag)))
    }

    pairs
  }

  def decide(headMorpheme: RankMorpheme, lastMorpheme: RankMorpheme): Boolean = {
    if (params.skipHeadAmbiguity == 2) headMorpheme.notAmbiguous()
    else if (params.skipHeadAmbiguity == 3) {
      headMorpheme.lessAmbiguous(lastMorpheme)
    }
    else true
  }

  def score(rankSequence: RankSequence, orders: Array[Int], prefix: String): RankSequence = {
    val sequence = rankSequence.rankMorphemes
    var iter = 0
    while (iter < params.maxRankIters) {
      orders.foreach(order => {
        sequence.sliding(order, 1).foreach(slice => {
          val headMorpheme = slice.head
          val lastMorpheme = slice.last
          if (decide(headMorpheme, lastMorpheme)) {
            val linkName = prefix + dependencyCategory(order)
            val pairs = pairize(headMorpheme, lastMorpheme, linkName)
            val totalConnections = pairs.length

            val totalLinkScore = pairs.filter { case (headTag, lastTag) => {
              val headVertex = mapNegative.getOrElse(headTag, Vertex("morph", headTag))
              val lastVertex = mapNegative.getOrElse(lastTag, Vertex("morph", lastTag))
              val score = headVertex.positive(lastVertex, "Negative")
              score == 0.0
            }}.map { case (headTag, lastTag) => {
              val headVertex = mapPositive.getOrElse(headTag, Vertex("morph", headTag))
              val lastVertex = mapPositive.getOrElse(lastTag, Vertex("morph", lastTag))
              val score = headVertex.positive(lastVertex, linkName)
              score
            }
            }.sum

            val avgLinkScore = totalLinkScore / (totalConnections + Double.MinPositiveValue)
            val rank = 0.10 * lastMorpheme.rank + 0.90 * avgLinkScore * headMorpheme.rank
            lastMorpheme.setRank(rank)
          }
        })
      })
      iter += 1
    }

    val totalRank = sequence.map(_.rank).sum
    rankSequence.setScore(totalRank)
  }

}
