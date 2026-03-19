package morphology.ranking

import morphology.data.{Label, MorphRanker, Params, Sentence}
import org.bytedeco.pytorch.{InputArchive, OutputArchive}
import torch.{Device, Float32, Tensor}
import torch.optim.SGD

import java.io.{File, FileInputStream, FileOutputStream, ObjectInputStream, ObjectOutputStream}
import scala.collection.SeqMap
import scala.util.Random


class MorphTrainer(params: Params) extends MorphRanker(params) {

  var contraintMap = Map[VertexPair, Double]()
  val (model, optimizer) = MorphTrainer.getModel(params)

  override def load(): MorphPredictor = {
    val modelFilename = params.rankModelFilename()
    val nnFilename = params.nmfModelFilename()
    if (new File(modelFilename).exists()) {
      println("Loading model")
      val objectStream = new ObjectInputStream(new FileInputStream(modelFilename))
      load(objectStream)
    }

    if (new File(nnFilename).exists()) {
      val archieve = new InputArchive()
      archieve.load_from(nnFilename)
      model.load(archieve)
    }
    this
  }

  override def save(): MorphPredictor = {
    println("Saving model...")
    val objectStream = new ObjectOutputStream(new FileOutputStream(params.rankModelFilename()))
    save(objectStream)
    objectStream.close()


    val archieve = new OutputArchive()
    model.save(archieve)
    archieve.save_to(params.nmfModelFilename())
    this
  }

  def size(): Int = {
    mapPositive.size
  }

  def update(src: Label, dst: Label): this.type = {
    val srcVertex = update("morph", src)
    val dstVertex = update("morph", dst)
    val pair = VertexPair(srcVertex, dstVertex)
    contraintMap = contraintMap.updated(pair, contraintMap.getOrElse(pair, 0d) + 1d)
    this
  }

  def constraint(src: Vertex, dst: Vertex): Double = {
    contraintMap(VertexPair(src, dst))
  }

  def constructMatrix(sequence: Array[RankMorpheme], orders: Array[Int], prefix: String): (Seq[Seq[Float]], Seq[Long]) = {

    var matrix = Seq.fill[Float](params.vocabSize, params.vocabSize)(0.0f)
    var indexes = Seq[Long]()
    orders.foreach(order => {
      sequence.sliding(order, 1).foreach(slice => {
        val headMorpheme = slice.head
        val lastMorpheme = slice.last
        if (decide(headMorpheme, lastMorpheme)) {
          val linkName = prefix + dependencyCategory(order)
          val pairs = pairize(headMorpheme, lastMorpheme, linkName)
          pairs.foreach { case (headTag, lastTag) => {
            val headVertex = mapPositive.getOrElse(headTag, Vertex("morph", headTag))
            val lastVertex = mapPositive.getOrElse(lastTag, Vertex("morph", lastTag))
            val headIndex = indexMap.getOrElse(headVertex, 0)
            val lastIndex = indexMap.getOrElse(headVertex, 0)
            val score = headVertex.positive(lastVertex, linkName)
            matrix = matrix.updated(headIndex, matrix(headIndex).updated(lastIndex,  score.toFloat))
            indexes = indexes :+ headIndex
            indexes = indexes :+ lastIndex
          }
          }
        }
      })
    })
    (matrix, indexes.distinct)
  }


  override def infer(results: Array[RankWord]): Map[Int, Array[RankMorpheme]] = {

    val items = results.map(_.suffixation())
    val morphemes = items.flatMap(rankWord=> rankWord.rankMorphemes)
    val (adjMatrix, indices) = constructMatrix(morphemes,params.wordSkipOrder, "Forward-")
    val scoreList = model.components(adjMatrix, indices, 0.09)
    var voteMap = (0 until items.length).map(i => (i -> Seq[RankMorpheme]())).toMap
    items.sliding(params.maxTokenWindow, 1).foreach { tokenWindow => {
      val windowCombinations = analyzer.combinatoricMorpheme(tokenWindow)
      windowCombinations.foreach(forwardPath => {
        val pathMorphemes = forwardPath.rankMorphemes.map(rankMorpheme=> {
          val total = rankMorpheme.labels.map(label=> {
            val vertex = Vertex("morph", label.tags)
            val idx = index(vertex)
            val idx_indice = indices.indexOf(idx)
            val idx_score =  (if idx_indice >= 0 then scoreList(idx_indice) else 0f)
            idx_score
          }).sum
          rankMorpheme.setRank(total / rankMorpheme.labels.length)
        })
        val total = pathMorphemes.map(morpheme=> morpheme.rank).sum
        forwardPath.setScore(total)

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
    }
    }


    val shuffleMap = voteMap.view.mapValues(s => {
      Random.shuffle(s).map(rankMorpheme => probScore(rankMorpheme))
    }).toMap

    val voteMax = shuffleMap.view.mapValues(pair => pair.sorted.toArray).toMap
    voteMax
  }

  override def construct(sentence: Sentence): this.type = {

    sentence.toNonStemLabels(params.maxSliceNgram).foreach(label => {
      if (probMap.contains(label)) probMap = probMap.updated(label, probMap(label) + 1d)
      else probMap = probMap.updated(label, 1d)
      count += 1
    })

    sentence.toSuffixPairs(params.maxPairAmbiguity, params.maxSliceNgram, linkMarker, params.wordSkipOrder).foreach {
      case (linkName, aTag, bTag) => {
        val aVertex = update("morph", aTag)
        val bVertex = update("morph", bTag)
        aVertex.add(linkName, bVertex, 1.0)
      }
    }

    val contraintPairs = sentence.toContraintPairs(params.maxSliceNgram)
    contraintPairs.foreach { case (src, dst) => {
      update(src, dst)
    }
    }

    this

  }

  override def train(): MorphTrainer = {
    var adjMatrix = Seq.fill[Float](params.vocabSize, params.vocabSize)(0.0f)
    var constraintMatrix = Array.ofDim[Float](params.vocabSize, params.vocabSize)
    val adjSize = adjMatrix.length
    mapPositive.values.toArray.foreach(src => {
      val srcIndex = indexMap(src)
      src.edges.toArray.foreach { case (key, score) => {
        val dst = mapPositive(key.value)
        val dstIndex = indexMap(dst)
        adjMatrix = adjMatrix.updated(
          srcIndex,
          adjMatrix(srcIndex).updated(dstIndex, score.toFloat)
        )
      }
      }
    })

    contraintMap.foreach { case (pair, score) => {
      val srcIndex = indexMap(pair.source)
      val dstIndex = indexMap(pair.destination)
      constraintMatrix = constraintMatrix.updated(
        srcIndex,
        constraintMatrix(srcIndex).updated(dstIndex, score.toFloat)
      )
    }
    }

    val A = Tensor[Float](adjMatrix).to(Device.CUDA)
    val loss = model.forward(A)
    print(f"\r###${loss.item}")
    loss.backward()
    optimizer.step()
    this
  }


}

object MorphTrainer {
  var singletonModel: NMFModel = null
  var optimizer: SGD = null

  def getModel(params: Params): (NMFModel, SGD) = {
    if singletonModel == null then {
      singletonModel = NMFModel(params).to(Device.CUDA)
      optimizer = SGD(
        singletonModel.parameters,
        lr = 0.1
      )
      (singletonModel, optimizer)
    }
    else {
      (singletonModel, optimizer)
    }

  }
}
