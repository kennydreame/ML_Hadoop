package classifier.bayes;

import com.google.common.collect.Sets;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FSDataOutputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.junit.Test;

import java.util.*;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

/**
 * Test cases
 */
public class ParallelRBNCUtilsTest {
  @Test
  public void testArraysXDWritable() throws Exception {
    Configuration conf = new Configuration();
    FileSystem fs = FileSystem.get(conf);

    // test 2D
    double[][] values = {{1, 2, 3}, {4, 5}, {}, {6}};
    ParallelRBNCUtils.Arrays2DWritable arrays2DWritable = new ParallelRBNCUtils.Arrays2DWritable(values);
    FSDataOutputStream output = fs.create(new Path("temp\\array2d.data"));
    arrays2DWritable.write(output);
    output.close();

    FSDataInputStream input = fs.open(new Path("temp\\array2d.data"));
    ParallelRBNCUtils.Arrays2DWritable arrays2D = new ParallelRBNCUtils.Arrays2DWritable();
    arrays2D.readFields(input);
    input.close();

    values = arrays2D.getValues();
    assertEquals(1, values[0][0], 1e-8);
    assertEquals(2, values[0][1], 1e-8);
    assertEquals(3, values[0][2], 1e-8);
    assertEquals(4, values[1][0], 1e-8);
    assertEquals(5, values[1][1], 1e-8);
    assertEquals(6, values[3][0], 1e-8);

    System.out.println(arrays2D);
  }

  @Test
  public void testArrays4DWritable() throws Exception {
    Configuration conf = new Configuration();
    FileSystem fs = FileSystem.get(conf);

    // test 4D
    double[][][][] values4D = {{{{1, 2}, {3}}, {}, {{4, 5}, {6, 7}, {}}}};
    ParallelRBNCUtils.Arrays4DWritable arrays4DWritable = new ParallelRBNCUtils.Arrays4DWritable(values4D);
    FSDataOutputStream output = fs.create(new Path("temp\\array2d.data"));
    arrays4DWritable.write(output);
    output.close();

    FSDataInputStream input = fs.open(new Path("temp\\array2d.data"));
    ParallelRBNCUtils.Arrays4DWritable arrays4D = new ParallelRBNCUtils.Arrays4DWritable();
    arrays4D.readFields(input);
    input.close();

    values4D = arrays4D.getValues();
    assertEquals(1, values4D[0][0][0][0], 1e-8);
    assertEquals(2, values4D[0][0][0][1], 1e-8);
    assertEquals(3, values4D[0][0][1][0], 1e-8);
    assertEquals(4, values4D[0][2][0][0], 1e-8);
    assertEquals(5, values4D[0][2][0][1], 1e-8);
    assertEquals(6, values4D[0][2][1][0], 1e-8);
    assertEquals(7, values4D[0][2][1][1], 1e-8);

    System.out.println(arrays4D);
  }

  @Test
  public void testModelStructure() throws Exception {
    Configuration conf = new Configuration();
    FileSystem fs = FileSystem.get(conf);
    List<Map<Integer, Set<Integer>>> structures = new ArrayList<Map<Integer, Set<Integer>>>();
    Map<Integer, Set<Integer>> component1 = new HashMap<Integer, Set<Integer>>();
    component1.put(0, Sets.newHashSet(2, 3));
    component1.put(1, Sets.newHashSet(2));
    component1.put(3, Sets.newHashSet(1, 0));
    Map<Integer, Set<Integer>> component2 = new HashMap<Integer, Set<Integer>>();
    component2.put(0, Sets.newHashSet(1, 3));
    component2.put(1, Sets.newHashSet(1, 2));
    component2.put(2, Sets.newHashSet(0, 1));
    structures.add(component1);
    structures.add(component2);
    ParallelRBNCUtils.writeModelStructures(fs, conf, new Path("temp\\temp1"), structures);
    System.out.println("************");
    structures = ParallelRBNCUtils.readModelStructures(fs, conf, new Path("temp\\temp1"));
    // first component
    Map<Integer, Set<Integer>> component = structures.get(0);
    System.out.println(component);
    assertEquals(component.size(), 3);
    assertArrayEquals(component.get(0).toArray(new Integer[]{}), new Integer[]{2, 3});
    assertArrayEquals(component.get(1).toArray(new Integer[]{}), new Integer[]{2});
    assertArrayEquals(component.get(3).toArray(new Integer[]{}), new Integer[]{0, 1});
    //second component
    component = structures.get(1);
    System.out.println(component);
    assertEquals(component.size(), 3);
    assertArrayEquals(component.get(0).toArray(new Integer[]{}), new Integer[]{1, 3});
    assertArrayEquals(component.get(1).toArray(new Integer[]{}), new Integer[]{1, 2});
    assertArrayEquals(component.get(2).toArray(new Integer[]{}), new Integer[]{0, 1});

    fs.delete(new Path("temp"), true);
  }
}
