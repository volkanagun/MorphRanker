package morphology.data

import morphology.ranking.{RankMorpheme, RankWord}

case class Word(index: Int, text: String, var originalAnalyses: Array[String]) {

  var predicted: String = null
  val splitBy = "[\\+\\>\\|]"

  def add(analyze: String): this.type = {
    this.originalAnalyses = originalAnalyses :+ analyze
    this
  }

  def analysisCount(): Int = originalAnalyses.length

  def notAmbiguous():Boolean = originalAnalyses.length==1

  def toNonStemLabels(): Set[Label] = {
    originalAnalyses.zipWithIndex.map(analysisPair => {
      val analysis = analysisPair._1
      val analysisIndex = analysisPair._2
      val tags = analysis.split("\\+")
      val stem = tags.head
      val analysisTag = tags.tail.mkString("+")
      val crrLabel = Label(index, text, stem, analysisTag, analysisTag)
        .setTotalCount(1)
        .setTrueLabel(analysisIndex)
        .setAmbiguousCount(originalAnalyses.length)

      crrLabel
    }).toSet
  }

  def toNonStemLabels(sliceSize: Int): Array[Label] = {
    originalAnalyses.zipWithIndex.flatMap(analysisPair => {
      val analysis = analysisPair._1
      val analysisIndex = analysisPair._2
      val tags = analysis.split("\\+")
      val stem = tags.head
      val analysisTag = tags.tail.mkString("+")
      val crrLabels = Label(index, text, stem, analysisTag, analysisTag)
        .setTotalCount(1)
        .setTrueLabel(analysisIndex)
        .setAmbiguousCount(originalAnalyses.length).slice(sliceSize)
      crrLabels
    })
  }

  def toLemmaLabels(): Array[Label] = {
    originalAnalyses.zipWithIndex.map(analysisPair => {
      val analysis = analysisPair._1
      val analysisIndex = analysisPair._2
      val tags = analysis.split("\\+")
      val stem = tags.head
      Label(index, text, stem, analysis, stem)
        .setLemmaLabel(true)
        .setTotalCount(1).setTrueLabel(analysisIndex)
        .setAmbiguousCount(originalAnalyses.length)
    })
  }

  def toDistinctLabels(sliceSize:Int):Array[Array[Label]] = {
    originalAnalyses.zipWithIndex.map{case(analysis,analysisIndex) => {
      val tags = analysis.split("\\+")
      val stem = tags.head
      val stemLabel = Label(index, text, stem, analysis, stem)
        .setAmbiguousCount(originalAnalyses.length)
        .setTrueLabel(analysisIndex)
        .setLemmaLabel(true)
        .setTotalCount(1)
      val tagLabels = tags.sliding(sliceSize, 1).map(_.mkString("+")).map(tagSeq=>{
        Label(index, text, stem, analysis,  tagSeq)
          .setAmbiguousCount(originalAnalyses.length)
          .setTrueLabel(analysisIndex)
          .setTotalCount(1)
      }).toArray
      val allLabels = stemLabel +: tagLabels
      allLabels
    }}
  }


  def toDirectLabels(): Set[Label] = {
    originalAnalyses.zipWithIndex.map(analysisPair => {
      val analysis = analysisPair._1
      val analysisIndex = analysisPair._2
      val analysisTag = analysis.split("\\+").mkString("+")
      val crrLabel = Label(index, text, text, analysisTag, analysisTag).setTotalCount(1).setAmbiguousCount(originalAnalyses.length)
        .setTrueLabel(analysisIndex)
      crrLabel
    }).toSet
  }

  def toStemLabels(sliceSize: Int): Array[Label] = {
    originalAnalyses.zipWithIndex.flatMap(analysisPair => {
      val analysis = analysisPair._1
      val analysisIndex = analysisPair._2
      val sliceTags = analysis.split("\\+").sliding(sliceSize, 1).map(_.mkString("+"))
      sliceTags.map(analysisTag => {
        Label(index, text, text, analysisTag, analysisTag).setTotalCount(1).setAmbiguousCount(originalAnalyses.length)
          .setTrueLabel(analysisIndex)
      })
    })
  }

  def toRankWord(sliceSize: Int): RankWord = {
    val rankMorphemes = originalAnalyses.zipWithIndex.map(analysisPair => {
      val analysis = analysisPair._1
      val analysisIndex = analysisPair._2
      val analysisSplit = analysis.split("\\+")
      val analysisLemma = analysisSplit.head
      var labelSlices = analysisSplit.tail.sliding(sliceSize, 1).map(_.mkString("+"))
        .map(analysisTag => {
          Label(index, text, analysisLemma, analysis, analysisTag)
            .setTotalCount(1)
            .setTrueLabel(analysisIndex)
            .setAmbiguousCount(originalAnalyses.length)
        }).toArray

      val lemmaLabel = Label(index, text, analysisLemma, analysis, analysisLemma)
        .setAmbiguousCount(originalAnalyses.length)
        .setTotalCount(1)
        .setLemmaLabel(true)

      labelSlices =  lemmaLabel +: labelSlices
      RankMorpheme(this.index, analysis, labelSlices, 0, 1d / originalAnalyses.length)
    })

    new RankWord(rankMorphemes)
  }

  def toUniqueSlices(slideSize: Int): Set[Label] = {
    val allAnalysis = toNonStemLabels().map(label => label.slice(slideSize))
    val distinctLabels = allAnalysis.zipWithIndex.flatMap(crrAnalysisPair => {
      val crrAnalysis = crrAnalysisPair._1
      val otherSlices = allAnalysis.zipWithIndex.filter(otherAnalysisPair => otherAnalysisPair._2 != crrAnalysisPair._2)
        .flatMap(_._1)
      crrAnalysis.filter(crrSlice => !otherSlices.contains(crrSlice))
    })

    distinctLabels
  }

  def toUniqueAnalysisSlices(slideSize: Int): Array[Array[Label]] = {
    val allAnalysis = toNonStemLabels().map(label => label.slice(slideSize)).toArray
    val distinctLabels = allAnalysis.zipWithIndex.map(crrAnalysisPair => {
      val crrAnalysis = crrAnalysisPair._1
      val otherSlices = allAnalysis.zipWithIndex.filter(otherAnalysisPair => otherAnalysisPair._2 != crrAnalysisPair._2)
        .flatMap(_._1)
      crrAnalysis.filter(crrSlice => !otherSlices.contains(crrSlice))
    })

    distinctLabels
  }

  def withStemAnalysis(): Array[String] = {
    originalAnalyses.map(analysis => analysis.split(splitBy).mkString("+"))
  }

  def similarity(predictedAnalysis: String): Array[String] = {
    val predictionSplit = predictedAnalysis.split(splitBy)
    val scores = originalAnalyses.map(originalAnalyze => {
      val originalLength = originalAnalyze.split(splitBy).length
      val splittedPredictionTagLength = predictionSplit.map(subarray => subarray.length).sum
      val originalMatchCount = predictionSplit.filter(crrTag => {
        originalAnalyze.contains(crrTag)
      }).length

      val size = (originalLength * splittedPredictionTagLength)
      (originalAnalyze, originalMatchCount.toDouble / size)
    }).sortBy(_._2)


    scores.map(_._1)
  }


}