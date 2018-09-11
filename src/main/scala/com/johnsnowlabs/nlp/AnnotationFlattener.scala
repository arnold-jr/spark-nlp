package com.johnsnowlabs.nlp

import org.apache.spark.ml.Transformer
import org.apache.spark.ml.param.{Param, ParamMap}
import org.apache.spark.ml.util.{DefaultParamsReadable, DefaultParamsWritable, Identifiable}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import org.apache.spark.sql._
import org.apache.spark.sql.expressions.UserDefinedFunction

/**
  * Created by arnold-jr on 09/11/18.
  */


class AnnotationFlattener(override val uid: String)
  extends Transformer
    with DefaultParamsWritable
    with HasAnnotatorType
    with HasOutputAnnotationCol {

  import com.johnsnowlabs.nlp.AnnotatorType._

  private type DocumentationContent = Row

  val inputCol: Param[String] = new Param[String](this, "inputCol", "input text column for processing")

  val idCol: Param[String] = new Param[String](this, "idCol", "id column for row reference")

  val metadataCol: Param[String] = new Param[String](this, "metadataCol", "metadata for document column")

  setDefault(outputCol, DOCUMENT)

  override val annotatorType: AnnotatorType = TOKEN

  def setInputCol(value: String): this.type = set(inputCol, value)

  def getInputCol: String = $(inputCol)

  def setIdCol(value: String): this.type = set(idCol, value)

  def getIdCol: String = $(idCol)

  def setMetadataCol(value: String): this.type = set(metadataCol, value)

  def getMetadataCol: String = $(metadataCol)

  def this() = this(Identifiable.randomUID("tokenselect"))

  override def copy(extra: ParamMap): Transformer = defaultCopy(extra)

  private def assemble(text: String, metadata: Map[String, String]): Seq[Annotation] = {
    Seq(Annotation(annotatorType, 0, text.length - 1, text, metadata))
  }

  private def dfAssembleNoExtras: UserDefinedFunction = udf {
    (text: String) =>
      assemble(text, Map.empty[String, String])
  }

  /** requirement for pipeline transformation validation. It is called on fit() */
  override final def transformSchema(schema: StructType): StructType = {
    val metadataBuilder: MetadataBuilder = new MetadataBuilder()
    metadataBuilder.putString("annotatorType", annotatorType)
    val outputFields = schema.fields :+
      StructField(getOutputCol, ArrayType(StringType), nullable = false, metadataBuilder.build)
    StructType(outputFields)
  }


  override def transform(dataset: Dataset[_]): DataFrame = {

    val metadataBuilder: MetadataBuilder = new MetadataBuilder()
    metadataBuilder.putString("annotatorType", annotatorType)

    dataset.withColumn(
      getOutputCol,
      Annotation.flattenArray(col(getInputCol)).as(getOutputCol, metadataBuilder.build)
    )
  }

}

object AnnotationFlattener extends DefaultParamsReadable[AnnotationFlattener]

