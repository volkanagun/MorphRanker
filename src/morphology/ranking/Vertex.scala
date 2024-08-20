package morphology.ranking

import java.io.{ObjectInputStream, ObjectOutputStream}


case class Vertex(var label: String, var value: String) extends Serializable {

  var edges: Map[VertexKey, Double] = Map()
  var isLemma = false
  var rank = 1d
  var rankCount = 0d
  var sum = 0d

  def setLabel(label: String): this.type = {
    this.label = label
    this
  }

  def setValue(value: String): this.type = {
    this.value = value
    this
  }

  def setLemma(isLemma:Boolean):this.type ={
    this.isLemma = isLemma
    this
  }

  def merge(other: Vertex): this.type = {
    other.edges.foreach { case (key, score) => {
      edges = edges.updated(key, edges.getOrElse(key, 0d) + score)
    }}

    this
  }

  def getRank():Double = rank

  def getAverageRank(): Double = {
    (this.rank+1) / (rankCount + 1)
  }

  def toKey(): VertexKey = {
    VertexKey(label, value)
  }

  def toKey(linkName: String): VertexKey = {
    VertexKey(linkName, value)
  }

  def save(outputStream: ObjectOutputStream): this.type = {

    outputStream.writeObject(label)
    outputStream.writeObject(value)

    outputStream.writeInt(edges.size)
    outputStream.writeDouble(rank)
    outputStream.writeDouble(sum)
    edges.filter(pair => !pair._1.equals(this)).foreach(pair => {
      outputStream.writeDouble(pair._2)
      outputStream.writeObject(pair._1)
    })

    this
  }

  def load(inputStream: ObjectInputStream): this.type = {

    label = inputStream.readObject().asInstanceOf[String]
    value = inputStream.readObject().asInstanceOf[String]
    val size = inputStream.readInt()
    rank = inputStream.readDouble()
    sum = inputStream.readDouble()

    Range(0, size).foreach(_ => {
      val count = inputStream.readDouble()
      val vertexKey = inputStream.readObject().asInstanceOf[VertexKey]
      edges = edges + (vertexKey -> count)
    })

    this
  }

  def setRank(rank: Double): this.type = {
    this.rank = rank
    this
  }

  def addRank(value: Double): this.type = {
    this.rank += value
    this.rankCount += 1
    this
  }

  def setEdges(edges: Map[VertexKey, Double]): this.type = {
    this.edges = edges
    this
  }

  def remove(vb: Vertex): this.type = {
    this.edges = this.edges.updated(vb.toKey(), 0d)
    this
  }

  def remove(vb: Vertex, linkName: String): this.type = {
    this.edges = this.edges.updated(vb.toKey(linkName), 0d)
    this
  }

  def setSum(sum: Double): this.type = {
    this.sum = sum
    this
  }


  def add(linkName: String, vertex: Vertex, value: Double): this.type = {
    val vertexKey = vertex.toKey(linkName)
    edges = edges.updated(vertexKey, edges.getOrElse(vertexKey, 0d) + value)
    this
  }

  def normalize(): this.type = {
    sum = edges.map(_._2).sum
    edges = edges.map { case (vertex, count) => (vertex, count / sum) }
    this
  }

  def normalizeByCategory(): this.type = {
    edges = edges.groupBy(_._1.value).flatMap {case(_, subedges)=>{
      val sum = subedges.map(_._2).sum
      val newedges = subedges.map { case (vertex, count) => (vertex, count / sum) }
      newedges
    }}

    this
  }

  def positive(target: Vertex, linkName: String): Double = {
    edges.getOrElse(target.toKey(linkName), 0d)
  }


  def negative(target: Vertex, linkName: String): Double = {
    edges.getOrElse(target.toKey(linkName), -1d) + 0.5d
  }

  def slimCopy(): Vertex = {
    Vertex("morph", value).setRank(rank)
  }

  def copy(): Vertex = {
    Vertex("morph", value).setRank(rank)
      .setEdges(edges)
  }
}
