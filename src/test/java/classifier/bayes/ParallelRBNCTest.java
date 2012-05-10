package classifier.bayes;

import com.google.common.collect.Lists;
import com.google.common.collect.Sets;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mrunit.mapreduce.MapDriver;
import org.apache.hadoop.mrunit.mapreduce.ReduceDriver;
import org.apache.hadoop.mrunit.types.Pair;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import java.util.*;

import static classifier.bayes.ParallelRBNCUtils.*;
import static classifier.bayes.ParallelRBNC.*;
import static org.junit.Assert.assertEquals;

/**
 * Test case for ParallelRBNC
 */
public class ParallelRBNCTest {
  private MapDriver<LongWritable, Text, LongWritable, Arrays4DWritable> mapDriver;

  private ReduceDriver<LongWritable, Arrays4DWritable, LongWritable, Arrays4DWritable> reduceDriver;

  Configuration conf;
  FileSystem fs;

  @Before
  public void setUp() throws Exception {
    mapDriver = new MapDriver<LongWritable, Text, LongWritable, Arrays4DWritable>(new ParallelCountingMapper());
    conf = mapDriver.getConfiguration();
    fs = FileSystem.get(conf);
    fs.mkdirs(new Path("temp"));
    conf.set("mapred.cache.localFiles", "./temp/dataset,./temp/structure");
    // dataset
    generateWekaFileHeader(5, "temp//dataset");
    // parents
    List<Map<Integer, Set<Integer>>> structure = new ArrayList<Map<Integer, Set<Integer>>>();
    Map<Integer, Set<Integer>> component1 = new HashMap<Integer, Set<Integer>>();
    component1.put(0, Sets.newHashSet(1, 2, 5));
    component1.put(1, Sets.newHashSet(2, 5));
    component1.put(2, Sets.newHashSet(5));
    component1.put(3, Sets.newHashSet(5));
    component1.put(4, Sets.newHashSet(5));
    component1.put(5, Sets.newHashSet(5));
    Map<Integer, Set<Integer>> component2 = new HashMap<Integer, Set<Integer>>();
    component2.put(0, Sets.newHashSet(1, 2, 5));
    component2.put(1, Sets.newHashSet(2, 5));
    component2.put(2, Sets.newHashSet(1, 5));
    component2.put(3, Sets.newHashSet(5));
    component2.put(4, Sets.newHashSet(2, 5));
    component2.put(5, Sets.newHashSet(5));
    structure.add(component1);
    structure.add(component2);
    ParallelRBNCUtils.writeModelStructures(fs, conf, new Path("temp//structure"), structure);
  }

  @After
  public void tearUp() throws Exception {
    fs.delete(new Path("temp"), true);
  }

  @Test
  public void testMapper() throws Exception {
    // test default
    List<Pair<LongWritable, ParallelRBNCUtils.Arrays4DWritable>> result = mapDriver.withInput(new LongWritable(10), new Text("1,0,1,0,0,1")).run();
    assertEquals(result.size(), 2);
    assertEquals(0L, result.get(0).getFirst().get());
    double[][][][] value0 = result.get(0).getSecond().getValues();
    assertEquals(value0[0][1][3][1], 1.0, 1e-8);
    assertEquals(value0[1][1][3][0], 1.0, 1e-8);
    assertEquals(value0[2][1][1][1], 1.0, 1e-8);
    assertEquals(value0[3][1][1][0], 1.0, 1e-8);
    assertEquals(value0[4][1][1][0], 1.0, 1e-8);
    assertEquals(value0[5][1][1][1], 1.0, 1e-8);

    assertEquals(1L, result.get(1).getFirst().get());
    double[][][][] value1 = result.get(1).getSecond().getValues();
    assertEquals(value1[0][1][3][1], 1.0, 1e-8);
    assertEquals(value1[1][1][3][0], 1.0, 1e-8);
    assertEquals(value1[2][1][1][1], 1.0, 1e-8);
    assertEquals(value1[3][1][1][0], 1.0, 1e-8);
    assertEquals(value1[4][1][3][0], 1.0, 1e-8);
    assertEquals(value1[5][1][1][1], 1.0, 1e-8);

    System.out.println(result.get(0).getSecond());
    System.out.println(result.get(1).getSecond());
  }

  @Test
  public void testReducer() throws Exception {
    reduceDriver = new ReduceDriver<LongWritable, ParallelRBNCUtils.Arrays4DWritable, LongWritable, ParallelRBNCUtils.Arrays4DWritable>(new ParallelCountingReducer());
    double[][][][] values4D1 = {{{{1, 2}, {3}}, {{4, 5}, {6, 7}}}};
    double[][][][] values4D2 = {{{{1, 2}, {3}}, {{4, 5}, {6, 7}}}};
    double[][][][] values4D3 = {{{{10, 20}, {30}}, {{40, 50}, {60, 70}}}};
    List<ParallelRBNCUtils.Arrays4DWritable> list = Lists.newArrayList(new ParallelRBNCUtils.Arrays4DWritable(values4D1));
    List<Pair<LongWritable, ParallelRBNCUtils.Arrays4DWritable>> result = reduceDriver.withInput(new LongWritable(0), list).run();
    double[][][][] values = result.get(0).getSecond().getValues();
    assertEquals(1, values[0][0][0][0], 1e-8);
    assertEquals(2, values[0][0][0][1], 1e-8);
    // multiple value
    list = Lists.newArrayList(new ParallelRBNCUtils.Arrays4DWritable(values4D1), new ParallelRBNCUtils.Arrays4DWritable(values4D2), new ParallelRBNCUtils.Arrays4DWritable(values4D3));
    result = reduceDriver.withInput(new LongWritable(0), list).run();
    values = result.get(0).getSecond().getValues();
    assertEquals(12, values[0][0][0][0], 1e-8);
    assertEquals(24, values[0][0][0][1], 1e-8);
    assertEquals(36, values[0][0][1][0], 1e-8);
    assertEquals(48, values[0][1][0][0], 1e-8);
    assertEquals(60, values[0][1][0][1], 1e-8);
    assertEquals(72, values[0][1][1][0], 1e-8);
    assertEquals(84, values[0][1][1][1], 1e-8);
  }
}
