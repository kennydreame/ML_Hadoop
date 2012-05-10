package classifier.bayes.standalone;

import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;
import weka.core.WeightedInstancesHandler;
import weka.estimators.DiscreteEstimator;
import weka.estimators.Estimator;

import java.util.ArrayList;
import java.util.List;

public class AnDE extends Classifier implements WeightedInstancesHandler {
	/** for serialization */
	public static final long serialVersionUID = -450879901061232190L;

	/** The number of class labels */
	private int m_NumClasses;

	/** n-Dependence */
	private int m_NDependence = 1;

	/** Store all the Component Classifiers in the ensemble */
	private List<Classifier> m_Classifiers = new ArrayList<Classifier>();

	@Override
	public void buildClassifier(Instances instances) throws Exception {
		// can classifier handle the data?
		getCapabilities().testWithFail(instances);

		// remove instances with missing class
		instances.deleteWithMissingClass();
		Instances data = new Instances(instances);

		m_NumClasses = instances.numClasses();
		switch (m_NDependence) {
		case 1:
			// construct AODE ensemble
			for (int att = 0; att < instances.numAttributes(); att++) {
				if (att != instances.classIndex()) {
					SuperParentNEstimators componetCls = new SuperParentNEstimators();
					componetCls.setSuperParentsAttIndex(new int[] { att });
					// build component classifier
					componetCls.buildClassifier(data);
					m_Classifiers.add(componetCls);
				}
			}
			break;
		case 2:
			// construct A2DE ensemble
			for (int att1 = 0; att1 < instances.numAttributes(); att1++)
				for (int att2 = att1 + 1; att2 < instances.numAttributes(); att2++) {
					if (att1 != instances.classIndex()
							&& att2 != instances.classIndex()) {
						SuperParentNEstimators componetCls = new SuperParentNEstimators();
						componetCls.setSuperParentsAttIndex(new int[] { att1,
								att2 });
						// build component classifier
						componetCls.buildClassifier(data);
						m_Classifiers.add(componetCls);
					}
				}
			break;
		default:
			System.out.println("No Classifier is construted");
		}
	}

	@Override
	public double[] distributionForInstance(Instance instance) throws Exception {
		double[] distpro = new double[m_NumClasses];
		for (Classifier cls : m_Classifiers) {
			double[] pro = cls.distributionForInstance(instance);
			for (int i = 0; i < pro.length; i++)
				distpro[i] += pro[i];
		}
		Utils.normalize(distpro);
		return distpro;
	}

	/**
	 * A Super Parent $n$-Dependence Estimator
	 */
	public static class SuperParentNEstimators extends Classifier implements
			WeightedInstancesHandler {

		/** for serialization */
		public static final long serialVersionUID = 1250389601216127217L;

		private int m_NumAttributes;

		private int m_NumClasses;

		private int m_ClassIndex;

		private int m_ClassSuperParentDenominator;

		private double m_SumOfInstances;

		private double[] m_ClassCounts;

		/** Super-parent attribute index for each attribute */
		private int[] m_SuperParentsAttIndex;

		/** the class and superparent value counts for each combination */
		private double[][] m_ClassSuperParentCounts;

		/** the CPTs of the n-Dependence Bayesian Network Classifier */
		private Estimator[][][] m_Distributions;

		/** The frequency of each attribute value for the dataset */
		private double[] m_Frequencies;

		/** An att's frequency must be this value or more to be a superParent */
		private int m_Limit = 1;

		/** flag for using m-estimates */
		private boolean m_Estimates = false;

		/** value for m in m-estimate */
		private double m_Weight = 1.0;

		/** flag for using Joint Probability Estimator of each component classifier */
		private boolean m_JointProbEstimate = true;

		@Override
		public void buildClassifier(Instances instances) throws Exception {
			// can classifier handle the data?
			getCapabilities().testWithFail(instances);

			// remove instances with missing class
			instances.deleteWithMissingClass();

			// set the values of the given variables
			m_NumAttributes = instances.numAttributes();
			m_NumClasses = instances.numClasses();
			m_ClassIndex = instances.classIndex();

			// initialize m_Distributions
			m_Distributions = new DiscreteEstimator[m_NumAttributes][][];

			// calculate the number of combination of superparents
			int parentsCardinality = 1;
			for (int iParent = 0; iParent < m_SuperParentsAttIndex.length; iParent++)
				parentsCardinality *= instances.attribute(
						m_SuperParentsAttIndex[iParent]).numValues();

			m_Frequencies = new double[parentsCardinality];
			m_ClassSuperParentCounts = new double[m_NumClasses][parentsCardinality];
			m_ClassSuperParentDenominator = m_NumClasses * parentsCardinality;

			for (int att = 0; att < m_NumAttributes; att++) {
				if (att == m_ClassIndex
						|| contains(m_SuperParentsAttIndex, att))
					continue;

				m_Distributions[att] = new DiscreteEstimator[m_NumClasses][parentsCardinality];
				for (int m = 0; m < m_NumClasses; m++)
					for (int n = 0; n < parentsCardinality; n++) {
						int numValues = instances.attribute(att).numValues();
						if (!m_Estimates) // using laplace estimation
							m_Distributions[att][m][n] = new DiscreteEstimator(
									numValues, true);
						else
							// using m-esitmation
							m_Distributions[att][m][n] =
							// new DiscreteEstimator( numValues, 0.5);
							new DiscreteEstimator(numValues, m_Weight
									/ numValues);
					}
			}

			m_ClassCounts = new double[m_NumClasses];
			// calculate for each instance
			for (int k = 0; k < instances.numInstances(); k++) {
				Instance inst = instances.instance(k);
				int classVal = (int) inst.classValue();
				m_SumOfInstances += inst.weight();
				m_ClassCounts[classVal] += inst.weight();

				int parentsValueIndex = 0;
				for (int iParent = 0; iParent < m_SuperParentsAttIndex.length; iParent++)
					parentsValueIndex = parentsValueIndex
							* inst.attribute(m_SuperParentsAttIndex[iParent])
									.numValues()
							+ (int) inst.value(m_SuperParentsAttIndex[iParent]);

				m_Frequencies[parentsValueIndex] += inst.weight();
				m_ClassSuperParentCounts[classVal][parentsValueIndex] += inst
						.weight();

				for (int att = 0; att < m_NumAttributes; att++) {
					if (att == m_ClassIndex
							|| contains(m_SuperParentsAttIndex, att))
						continue;

					m_Distributions[att][classVal][parentsValueIndex].addValue(
							inst.value(att), inst.weight());
				}
			}
		}

		@Override
		public double[] distributionForInstance(Instance inst) {
			double[] distpro = new double[m_NumClasses];

			int parentsValueIndex = 0;
			for (int iParent = 0; iParent < m_SuperParentsAttIndex.length; iParent++) {
				parentsValueIndex = parentsValueIndex
						* inst.attribute(m_SuperParentsAttIndex[iParent])
								.numValues()
						+ (int) inst.value(m_SuperParentsAttIndex[iParent]);
			}

			if (m_Frequencies[parentsValueIndex] < m_Limit)
				// zero estimate which is equivalent to omit this component
				// classifier's prediction
				return distpro;

			for (int classVal = 0; classVal < m_NumClasses; classVal++) {
				if (!m_Estimates) // using laplace estimation
					distpro[classVal] = (m_ClassSuperParentCounts[classVal][parentsValueIndex] + 1)
							/ (m_SumOfInstances + m_ClassSuperParentDenominator);
				else
					distpro[classVal] = (m_ClassSuperParentCounts[classVal][parentsValueIndex] + m_Weight
							/ m_ClassSuperParentDenominator)
							/ (m_SumOfInstances + m_Weight);

				for (int att = 0; att < m_NumAttributes; att++) {
					if (att == m_ClassIndex
							|| contains(m_SuperParentsAttIndex, att))
						continue;
					distpro[classVal] *= m_Distributions[att][classVal][parentsValueIndex]
							.getProbability(inst.value(att));
				}
			}

			if (!m_JointProbEstimate)
				Utils.normalize(distpro);

			return distpro;
		}

		/** Is the data array contains the given item */
		public static boolean contains(int[] data, int item) {
			for (int val : data)
				if (val == item)
					return true;
			return false;
		}

		/** Set and get some property */
		public int[] getSuperParentsAttIndex() {
			return m_SuperParentsAttIndex;
		}

		public void setSuperParentsAttIndex(int[] superParentsAttIndex) {
			m_SuperParentsAttIndex = superParentsAttIndex;
		}

		public boolean isMEstimates() {
			return m_Estimates;
		}

		public void setMEstimates(boolean estimates) {
			m_Estimates = estimates;
		}

		public boolean isJointProbEstimate() {
			return m_JointProbEstimate;
		}

		public void setJointProbEstimate(boolean jointProbEstimate) {
			m_JointProbEstimate = jointProbEstimate;
		}
	}

	/** get the NDependence */
	public int getNDependence() {
		return m_NDependence;
	}

	/** set the NDependence, default 1 (AODE) */
	public void setNDependence(int dependence) {
		m_NDependence = dependence;
	}
}
