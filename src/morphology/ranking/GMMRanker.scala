package morphology.ranking

import morphology.data.{Label, Params, Sentence}
import org.apache.commons.math3.distribution.MultivariateNormalDistribution
import org.apache.commons.math3.linear.{Array2DRowRealMatrix, ArrayRealVector, EigenDecomposition, MatrixUtils, RealMatrix}
import smile.data.DataFrame

import smile.math.MathEx.{cov, mean}
import smile.math.matrix.Matrix
import smile.projection.PCA
import smile.stat.distribution.MultivariateGaussianDistribution

import java.io.{ObjectInputStream, ObjectOutputStream}
import java.util.concurrent.ForkJoinPool
import scala.collection.parallel.CollectionConverters.{ArrayIsParallelizable, ImmutableIterableIsParallelizable, IterableIsParallelizable}
import scala.collection.parallel.ForkJoinTaskSupport
import scala.util.Random

class GMMRanker(params: Params) extends MorphPredictor(params) {

  var dictionary = Array[String]()
  var labelDictionary = Array[String]()
  var samples = Array[(Array[String], Array[String])]()

  //var gmm: Map[Int, (MultivariateNormalDistribution, RealMatrix)] = _
  var gmm: Map[Int, (MultivariateGaussianDistribution, PCA)] = _

  override def merge(other: MorphPredictor): GMMRanker.this.type = {
    val otherRanker = other.asInstanceOf[GMMRanker]
    samples = samples ++ otherRanker.samples
    otherRanker.labelDictionary.foreach(item => {
      if (!labelDictionary.contains(item)) {
        labelDictionary :+= item
      }
    })
    otherRanker.dictionary.foreach(feature => {
      if (!dictionary.contains(feature)) {
        dictionary :+= feature
      }
    })
    this
  }

  def multivariateMean(array: Array[Array[Double]]): Array[Double] = {
    println("Computing mean...")
    val d = array.head.length
    val mean = Array.fill(d)(0d)
    (0 until d).par.foreach(i => {
      mean(i) = mean(i) + array.par.map(item => item(i)).sum / array.length
    })
    mean
  }

  def meanSmile(array: Array[Array[Double]]): Array[Double] = {
    println("Computing mean...")
    val df = DataFrame.of(array)
    val means = (0 until df.ncols()).toArray.map(i => mean(df.column(i).toDoubleArray))
    means
  }


  def centerDataRow(data: Array[Double], mean: Array[Double]): Array[Double] = {
    data.zip(mean).map { case (x, m) => x - m }
  }

  def centerData(data: Array[Array[Double]], mean: Array[Double]): Array[Array[Double]] = {
    data.map(centerDataRow(_, mean))
  }

  def multivariateCov(data: Array[Array[Double]], mean: Array[Double]): Array[Array[Double]] = {
    println("Covariance computation...")
    val covarianceMatrix = cov(data, mean)
    covarianceMatrix
  }

  def covSmile(data: Array[Array[Double]], mean: Array[Double]): Array[Array[Double]] = {
    println("Covariance computation...")
    val covarianceMatrix = cov(data, mean)
    covarianceMatrix
  }

  def regulizedCov(matrix: Array[Array[Double]]): Array[Array[Double]] = {
    val d = matrix.length
    (0 until d).par.foreach { i => matrix(i)(i) = (100 + Random.nextGaussian()) * 1E-5 }
    matrix
  }

  def pca(covarianceMatrix: Array[Array[Double]]): RealMatrix = {
    println("Computing pca")
    val regularized = regularizeMatrix(covarianceMatrix)
    val realMatrix = new Array2DRowRealMatrix(regularized)

    val eigenDecomposition = new EigenDecomposition(realMatrix)
    val eigenVectors = eigenDecomposition.getV
    val vectors = eigenVectors.getData.take(params.kDims)
    val projectionMatrix = new Array2DRowRealMatrix(vectors).transpose()
    println("PCA computation is finished...")
    projectionMatrix
  }

  def pcaSmile(covarianceMatrix: Array[Array[Double]]): PCA = {
    println("Computing pca")
    val pcaModel = PCA.fit(covarianceMatrix)
    println("PCA computation is finished...")
    pcaModel.setProjection(params.kDims)
    pcaModel
  }

  def project(data: Array[Array[Double]], projectionMatrix: RealMatrix): Array[Array[Double]] = {
    val matrix = new Array2DRowRealMatrix(data)
    matrix.multiply(projectionMatrix).getData
  }

  def projectSmile(data: Array[Array[Double]], pca: PCA): Array[Array[Double]] = {
    pca.project(data)
  }

  def projectVector(data: Array[Double], projectionMatrix: RealMatrix): Array[Double] = {
    val matrix = new Array2DRowRealMatrix(data).transpose()
    matrix.multiply(projectionMatrix).getData.head
  }

  def regularizeMatrix(covMatrix: Array[Array[Double]]): Array[Array[Double]] = {
    val mainMatrix = new Array2DRowRealMatrix(covMatrix)
    val identityMatrix = MatrixUtils.createRealIdentityMatrix(mainMatrix.getRowDimension)
    val regularizationFactor = 0.01
    mainMatrix.add(identityMatrix.scalarMultiply(regularizationFactor))
      .getData
  }

  def regularizeMatrixSmile(covMatrix: Array[Array[Double]]): Array[Array[Double]] = {

    val n = covMatrix.length
    val m = covMatrix.head.length
    val covarianceMatrix = new Matrix(covMatrix)
    val regularizationFactor = 0.01
    val identityMatrix = Matrix.eye(n, m).mul(regularizationFactor)
    covarianceMatrix.add(identityMatrix).toArray

  }

  def multivariateGaussian(map: Map[Int, Array[Array[Double]]]): Unit = {
    println("Total target tags: " + map.size)
    val parKeys = map.keys.par

    println("Constructing multivariate gaussian")
    parKeys.tasksupport = new ForkJoinTaskSupport(new ForkJoinPool(24))
    gmm = parKeys.flatMap(i => {
      try {
        println("Current tag: " + labelDictionary(i))
        val ds = map(i)
        //val mean = multivariateMean(ds)
        //val cov = multivariateCov(ds, mean)
        //val paa = pca(ds)
        val paa = pcaSmile(ds)
        val newdata = projectSmile(ds, paa)
        val newmean = meanSmile(newdata)
        val newcov = covSmile(newdata, newmean)
        val regcov = regularizeMatrixSmile(newcov)
        val regcovMatrix = new Matrix(regcov)
        println(s"Symmetric for index " + i + ":" + regcovMatrix.isSymmetric)
        val distribution = new MultivariateGaussianDistribution(newmean, regcovMatrix)
        println("Computing gaussian finished for index " + i)
        Some(i -> (distribution, paa))
      }
      catch {
        case ex: Exception => {
          ex.printStackTrace()
          None
        }
      }
    }).toArray.toMap
  }

  def constructStats(dataBy: Map[Int, Array[Array[Double]]]): Unit = {
    println("Constructing statistics")
    val totalSamples = dataBy.map(pair => pair._2.length).sum
    val totalTargetTags = dataBy.size

    params.trainStats.totalSentences = totalSamples
    params.trainStats.totalTags = totalTargetTags

  }

  def fit(dataBy: Map[Int, Array[Array[Double]]]): Unit = {

    constructStats(dataBy)
    multivariateGaussian(dataBy)
  }

  def train(): this.type = {
    println("Training ")
    val vectors = construct()

    reset()
    fit(vectors)
    this
  }

  def reset(): this.type = {
    samples = Array()
    System.gc()
    this
  }

  def addSample(sentence: Array[RankWord]): MorphPredictor = {
    val crrSample = sentence.sliding(params.maxTokenWindow, 1).toArray.par.map(windowSequence => {
      analyzer.combinatoricMorpheme(windowSequence)
    }).toArray.flatMap(windows => {
      windows.flatMap(rankSequence => {
        getAllFeatures(rankSequence.rankMorphemes)
      })
    })

    samples = samples ++ crrSample
    //println("Sample added...")

    this
  }

  def getDistance(distance: Int): String = {
    val prefix = if (distance < 0) "Backward-" else "Forward-"
    prefix + (if (distance > 5) "Distant" else if (distance >= 1) "Neighbour" else "Local")
  }

  def getFeatures(sentence: Array[RankMorpheme]): Array[(Array[String], Array[String])] = {
    val analysisSequence = sentence.map(rankMorpheme => rankMorpheme.labels)
    var bow = Array[(Array[String], Array[String])]()

    for (i <- 0 until analysisSequence.length) {
      val targetLabels = analysisSequence(i)
      if (targetLabels.nonEmpty) {
        val ll = Math.max(i - params.maxTokenWindow, 0)
        val rr = Math.min(i + params.maxTokenWindow, analysisSequence.length)
        var crrFeatures = Set[String]()

        for (w <- ll until rr) {
          val wTag = "@" + getDistance(i - w)
          val wFeature = analysisSequence(w)
          if (wFeature.nonEmpty) {
            val wFeatures = wFeature.map(wLabel => wLabel.tags + wTag)
            crrFeatures ++= wFeatures
          }
        }

        val crrTargets = targetLabels.map(targetLabel => {
          targetLabel.tags
        }).toSet

        bow :+= (crrFeatures.toArray, crrTargets.toArray)
      }
    }
    bow
  }

  def getAllFeatures(sentence: Array[RankMorpheme]): Array[(Array[String], Array[String])] = {
    val analysisSequence = sentence.map(rankMorpheme => rankMorpheme
      .labels
      .filter(!_.lemmaLabel))

    var bow = Array[(Array[String], Array[String])]()

    for (i <- 0 until sentence.length) {
      val targetMorpheme = sentence(i)
      if (targetMorpheme.notAmbiguous()) {
        val targetLabels = analysisSequence(i)
        val ll = Math.max(i - params.maxTokenWindow, 0)
        val rr = Math.min(i + params.maxTokenWindow, analysisSequence.length)
        var crrFeatures = Set[String]()

        for (w <- ll until rr) {
          val wTag = "@" + getDistance(i - w)
          val wFeature = analysisSequence(w)
          if (wFeature.nonEmpty) {
            val wFeatures = wFeature.map(wLabel => wLabel.tags + wTag)
            crrFeatures ++= wFeatures
          }
        }

        val crrTargets = targetLabels.map(targetLabel => {
          targetLabel.tags
        }).toSet

        bow :+= (crrFeatures.toArray, crrTargets.toArray)
      }
    }
    bow
  }

  def constructFeatures(sample: Array[(Array[String], Array[String])]): Unit = {
    val keySet = sample.flatMap(_._2).toSet
    val map = keySet.map(key => {
      val featureArray = sample.filter(pair => {
        pair._2.contains(key)
      }).map(pair => {
        pair._1
      })

      key -> featureArray
    }).toMap

    constructFeatures(map)
  }

  def constructFeatures(features: Map[String, Array[Array[String]]]): Unit = {
    features.foreach(item => {
      val crrTag = item._1
      val crrFeatures = item._2.flatten.toSet
      crrFeatures.foreach(item => {
        if (isTraining && !dictionary.contains(item)) {
          dictionary :+= item
        }
      })
      if (isTraining) {
        labelDictionary :+= crrTag
      }
    })
  }

  def filter(): Map[String, Array[Array[String]]] = {
    println("Filtering")
    val tagSet = samples.flatMap(_._2).toSet.toArray
    val tagMap = tagSet.par.map(tag => tag -> samples.filter(pairs => pairs._2.contains(tag)).take(params.tagSamples).map(_._1))
      .toArray
      .toMap

    samples = Array()
    System.gc()
    tagMap
  }

  def construct(): Map[Int, Array[Array[Double]]] = {
    println("Constructing features")
    val sampleMap = filter()
    constructFeatures(sampleMap)
    println("Vectorizing features")
    vectorize(sampleMap)
  }

  def vectorize(sampleMap: Map[String, Array[Array[String]]]): Map[Int, Array[Array[Double]]] = {

    sampleMap.par.map(item => {
      val crrTag = labelDictionary.indexOf(item._1)
      val crrSamples = item._2
      val vectors = crrSamples.map(crrBOWFeatures => {
        val featureVector = Array.fill[Double](params.maxFeatures)(0d)

        crrBOWFeatures.foreach(crrFeature => {
          val index = dictionary.indexOf(crrFeature)
          if (index >= 0) featureVector(index) = 1.0
        })

        featureVector
      })
      crrTag -> vectors
    }).toArray.toMap
  }

  def vectorize(samples: Array[(Array[String], Array[String])]): Array[(Array[Double], Array[Int])] = {

    samples.map(item => {
      val crrFeatures = item._1
      val crrTags = item._1
      val crrVector = Array.fill[Double](params.maxFeatures)(0d)
      crrFeatures.foreach(crrFeature => {
        val index = dictionary.indexOf(crrFeature)
        if (index >= 0) crrVector(index) = 1.0
      })
      val tagIndices = crrTags.map(crrTag => dictionary.indexOf(crrTag)).filter(_ >= 0)

      (crrVector, tagIndices)
    })
  }

  override def save(objectStream: ObjectOutputStream): MorphPredictor = {

    objectStream.writeInt(gmm.size)
    gmm.foreach { case (i, pair) => {
      objectStream.writeInt(i)
      //val mean = pair._1.getMeans
      //val cov = pair._1.getCovariances.getData
      //val project = pair._2.getData
      val mean = pair._1.mean()
      val cov = pair._1.cov().toArray
      val project = pair._2

      objectStream.writeObject(mean)
      objectStream.writeObject(cov)
      objectStream.writeObject(project)
    }
    }

    objectStream.writeObject(dictionary)
    objectStream.writeObject(labelDictionary)

    this
  }

  override def load(objectStream: ObjectInputStream): MorphPredictor = {
    val size = objectStream.readInt()
    gmm = (0 until size).map(_ => {
      val key = objectStream.readInt()
      val mean = objectStream.readObject().asInstanceOf[Array[Double]]
      val cov = objectStream.readObject().asInstanceOf[Array[Array[Double]]]
      //val project = objectStream.readObject().asInstanceOf[Array[Array[Double]]]
      val project = objectStream.readObject().asInstanceOf[PCA]

      //val gaussian = new MultivariateNormalDistribution(mean, cov)
      val gaussian = new MultivariateGaussianDistribution(mean, new Matrix(cov))
      //key -> (gaussian, projectMatrix)
      key -> (gaussian, project)
    }).toMap

    dictionary = objectStream.readObject().asInstanceOf[Array[String]]
    labelDictionary = objectStream.readObject().asInstanceOf[Array[String]]

    this
  }

  def predict(dataVector: Array[Double], keys: Array[Int]): Array[(Int, Double)] = {
    keys.filter(i => gmm.contains(i)).map(i => {
      val (normal, paa) = gmm(i)
      // i -> normal.density(projectVector(dataVector, paa))
      val projectionVector = paa.project(dataVector)
      val gaussian = normal.p(projectionVector)
      i -> gaussian
    }).filter(_._2 > params.kThreshold)
  }

  def predict(data: Array[(Array[Double], Array[Int])]): Array[Array[(Int, Double)]] = {

    data.map { x => {
      val dataVector = x._1
      val keyVector = x._2
      predict(dataVector, keyVector)
    }
    }
  }

  override def infer(words: Array[RankWord]): Map[Int, Array[RankMorpheme]] = {
    var voteMax = (0 until words.length).map(i => i -> Array[RankMorpheme]()).toMap

    words.sliding(params.maxTokenWindow, 1).foreach(crrWindow => {
      val rankSequences = analyzer.combinatoricMorpheme(crrWindow)
      rankSequences.zipWithIndex.foreach { case (rankSequence, _) => {
        val rankMorphemes = rankSequence.rankMorphemes
        val vectorPairs = vectorize(getFeatures(rankMorphemes))
        val predictionLabels = predict(vectorPairs)
        var score = 0d

        rankMorphemes.zip(predictionLabels).foreach { case (rankMorpheme, indices) => {
          val rank = indices.map(_._2).sum
          var targetMorphemes = voteMax.getOrElse(rankMorpheme.tokenIndex, Array())
          val morphemeIndex = targetMorphemes.indexOf(rankMorpheme)
          if (morphemeIndex != -1) {
            val targetMorpheme = targetMorphemes(morphemeIndex)
            targetMorpheme.addRank(rank)
          }
          else {
            targetMorphemes :+= rankMorpheme
              .addRank(rank)
          }

          voteMax = voteMax.updated(rankMorpheme.tokenIndex, targetMorphemes)
          score += indices.length
        }
        }
      }
      }

    })

    val voteMap = voteMax.view.mapValues { array => array.sorted}.toMap
    voteMap
  }

  override def construct(sequence: Sentence): MorphPredictor = {
    val rankWords = sequence.words.map(_.toRankWord(params.maxSliceNgram))
    addSample(rankWords)
  }

  override def trigger(): MorphPredictor = train()

}