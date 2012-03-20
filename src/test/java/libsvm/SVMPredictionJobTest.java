package libsvm;

import libsvm.libsvm.svm_node;
import libsvm.mapreduce.SVMPredictionMapper;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mrunit.mapreduce.MapDriver;
import org.apache.hadoop.mrunit.types.Pair;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.junit.Before;
import org.junit.Test;

import java.util.List;

import static libsvm.mapreduce.SVMPredictionJob.parseInstance;
import static org.junit.Assert.assertEquals;

/**
 * Test case for SVMPrediction
 */
public class SVMPredictionJobTest {
  private Configuration conf;
  private FileSystem fs;
  private MapDriver<Text, VectorWritable, Text, DoubleWritable> mapDriver;

  @Before
  public void setUp() throws Exception {
    SVMPredictionMapper mapper = new SVMPredictionMapper();
    mapDriver = new MapDriver<Text, VectorWritable, Text, DoubleWritable>(mapper);

    conf = mapDriver.getConfiguration();
    fs = FileSystem.getLocal(conf);
    conf.set("mapred.cache.localFiles", "./libsvm/heart_scale.model");
  }

  @Test
  public void testMapper() throws Exception {
    Vector vector = new DenseVector(new double[]{0.708333, 1, 1, -0.320755, -0.105023, -1, 1, -0.419847, -1, -0.225806, 0, 1, -1});
    List<Pair<Text, DoubleWritable>> result = mapDriver.withInput(new Text("002011id"), new VectorWritable(vector)).run();
    assertEquals(result.size(), 1);
    assertEquals(result.get(0).getSecond().get(), -1.0, 1e-8);

    vector = new DenseVector(new double[]{0.291667, 1, 1, -0.132075, -0.237443, -1, 1, 0.51145, -1, -0.612903, 0, 0.333333, 1});
    vector = new RandomAccessSparseVector(vector);
    result = mapDriver.withInput(new Text("002011id"), new VectorWritable(vector)).run();
    assertEquals(result.size(), 1);
    assertEquals(result.get(0).getSecond().get(), 1.0, 1e-8);
  }

  @Test
  public void testParseInstance() throws Exception {
    String line = " 1 1:-0.2 2:1 3:1 4:-0.1 7:-0.5";
    Vector vector = new DenseVector(new double[]{-0.2, 1, 1, -0.1, 0, 0, -0.5});
    svm_node[] x = parseInstance(line, false);
    svm_node[] y = parseInstance(vector);
    assertEquals(x.length, 5);
    assertEquals(x[0].index, 1);
    assertEquals(x[0].value, -0.2, 1e-8);
    assertEquals(x[1].index, 2);
    assertEquals(x[1].value, 1, 1e-8);
    assertEquals(x[2].index, 3);
    assertEquals(x[2].value, 1, 1e-8);
    assertEquals(x[3].index, 4);
    assertEquals(x[3].value, -0.1, 1e-8);
    assertEquals(x[4].index, 7);
    assertEquals(x[4].value, -0.5, 1e-8);
    assertEquals(x.length, y.length);
    for (int i = 0; i < x.length; i++) {
      assertEquals(x[i].index, y[i].index);
      assertEquals(x[i].value, y[i].value, 1e-8);
    }
    // test first item is id
    line = "id 1 1:-0.2 2:1 3:1 4:-0.1 7:-0.5";
    x = parseInstance(line, true);
    assertEquals(x.length, 5);
    assertEquals(x[0].index, 1);
    assertEquals(x[0].value, -0.2, 1e-8);
    assertEquals(x[1].index, 2);
    assertEquals(x[1].value, 1, 1e-8);
    assertEquals(x[2].index, 3);
    assertEquals(x[2].value, 1, 1e-8);
    assertEquals(x[3].index, 4);
    assertEquals(x[3].value, -0.1, 1e-8);
    assertEquals(x[4].index, 7);
    assertEquals(x[4].value, -0.5, 1e-8);
  }
}
