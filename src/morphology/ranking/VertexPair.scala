package morphology.ranking

class VertexPair(val source:Vertex, val destination:Vertex) {
  override def hashCode(): Int = source.hashCode() * 7 + destination.hashCode()

  override def equals(obj: Any): Boolean = {
    val other = obj.asInstanceOf[VertexPair]
    other.source == source && other.destination == destination
  }
}
