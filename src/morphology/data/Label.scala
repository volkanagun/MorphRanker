package morphology.data

case class Label(index: Int, word: String, stem: String, analysis: String, var tags: String) {

  var totalCount = 0
  var ambiguousCount = 0
  var trueLabel = false
  var lemmaLabel = false
  var analysisIndex = 0


  override def hashCode(): Int = tags.hashCode()

  def sparseCode(): Int = {
    var result = tags.hashCode
    result = result * 7 + index
    result = result * 7 + analysisIndex
    result
  }

  override def equals(obj: Any): Boolean = tags.equals(obj.asInstanceOf[Label].tags)

  def setTrueLabel(analysisIndex: Int): this.type = {
    this.trueLabel = analysisIndex == 0
    this
  }

  def setLemmaLabel(isLabel: Boolean): this.type = {
    this.lemmaLabel = isLabel
    this
  }

  def setTrueLabel(trueLabel: Boolean): this.type = {
    this.trueLabel = trueLabel
    this
  }

  def setTotalCount(totalCount: Int): this.type = {
    this.totalCount = totalCount
    this
  }

  def setAnalysisIndex(analysisIndex: Int): this.type = {
    this.analysisIndex = analysisIndex
    this
  }

  def setAmbiguousCount(ambiguousCount: Int): this.type = {
    this.ambiguousCount = ambiguousCount
    this
  }

  def suffixation(): this.type = {

    val suffix = (if word.length >= stem.length then word.substring(stem.length) else stem)
    if suffix.nonEmpty && !tags.startsWith(suffix) then
      this.tags = suffix + ":" + tags
    this
  }

  def mege(other: Label): this.type = {
    this.totalCount += other.totalCount
    this.ambiguousCount += other.ambiguousCount
    this
  }

  def ambiguousScore(): Double = {
    ambiguousCount.toDouble / totalCount
  }

  def notAmbiguous(): Boolean = {
    ambiguousScore() == 1.0
  }

  def toAnalysisLabel(): Label = {
    Label(index, word, stem, analysis, analysis)
  }

  def slice(range: Int): Array[Label] = {

    tags.split("\\+").sliding(range, 1).map(tagSlice => {
      Label(index, word, stem, tags, tagSlice.mkString("+"))
        .setTotalCount(totalCount)
        .setTrueLabel(trueLabel)
        .setAmbiguousCount(ambiguousCount)
    }).toArray
  }

  override def toString: String = tags
}
