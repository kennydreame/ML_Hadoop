package classifier.bayes.standalone;

import weka.classifiers.Classifier;
import weka.classifiers.bayes.BayesNet;
import weka.classifiers.bayes.net.search.SearchAlgorithm;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;

import java.util.*;

public class RandomBayesNetClassifiers extends Classifier {
	/** for serialization */
	private static final long serialVersionUID = 2181987877484697882L;

	/** Holds upper bound on number of parents */
	private int m_nMaxNrOfParents = 2;

	/**
	 * whether initial structure is an empty graph or a Naive Bayes network
	 */
	private boolean m_bInitAsNaiveBayes = true;

	/** Holds flag to indicate ordering should be random * */
	private boolean m_bRandomOrder = true;

	/** Number of class labels */
	private int m_ClassNum;

	/** ensemble size */
	private int m_Size = 20;

	/** Collection of multiple random generate Bayes Network Classifiers */
	private List<Classifier> m_Ensembles = new ArrayList<Classifier>();;

	@Override
	public void buildClassifier(Instances data) throws Exception {
		// remove instances with missing class
		data = new Instances(data);
		data.deleteWithMissingClass();

		m_ClassNum = data.numClasses();

		for (int i = 0; i < m_Size; i++) {
			BayesNet componetBayesNet = new BayesNet();
			// config the RandomSearch algorithm
			RandomSearch componentSA = new RandomSearch();
			componentSA.setRandomOrder(m_bRandomOrder);
			componentSA.setMaxNrOfParents(m_nMaxNrOfParents);
			componentSA.setInitAsNaiveBayes(m_bInitAsNaiveBayes);
			componentSA.setSeed(i);
			// 
			componetBayesNet.setSearchAlgorithm(componentSA);
			componetBayesNet.buildClassifier(data);
			m_Ensembles.add(componetBayesNet);
		}
	}

	@Override
	public double[] distributionForInstance(Instance instance) throws Exception {
		double[] distpro = new double[m_ClassNum];
		label: for (int index = 0; index < m_Size; index++) {
			Classifier classifier = m_Ensembles.get(index);
			double[] pro = classifier.distributionForInstance(instance);
			for (int i = 0; i < pro.length; i++)
				if (new Double(pro[i]).equals(Double.NaN)) {
					System.err.println("һ��component classifier��Ԥ�����ֵΪNaN");
					continue label;
				}
			// sum the pro
			for (int i = 0; i < distpro.length; i++)
				distpro[i] += pro[i];
		}
		Utils.normalize(distpro);
		return distpro;
	}

	/**
	 * Sets the number of component classifiers
	 * 
	 * @param size
	 */
	public void setSize(int size) {
		m_Size = size;
	}

	/**
	 * get the number of component classifiers
	 * 
	 * @param size
	 */
	public void getSize(int size) {
		m_Size = size;
	}

	/**
	 * Sets the max number of parents
	 * 
	 * @param nMaxNrOfParents
	 *            the max number of parents
	 */
	public void setMaxNrOfParents(int nMaxNrOfParents) {
		m_nMaxNrOfParents = nMaxNrOfParents;
	}

	/**
	 * Gets the max number of parents.
	 * 
	 * @return the max number of parents
	 */
	public int getMaxNrOfParents() {
		return m_nMaxNrOfParents;
	}

	/**
	 * Sets whether to init as naive bayes
	 * 
	 * @param bInitAsNaiveBayes
	 *            whether to init as naive bayes
	 */
	public void setInitAsNaiveBayes(boolean bInitAsNaiveBayes) {
		m_bInitAsNaiveBayes = bInitAsNaiveBayes;
	}

	/**
	 * Gets whether to init as naive bayes
	 * 
	 * @return whether to init as naive bayes
	 */
	public boolean getInitAsNaiveBayes() {
		return m_bInitAsNaiveBayes;
	}

	/**
	 * Set random order flag
	 * 
	 * @param bRandomOrder
	 *            the random order flag
	 */
	public void setRandomOrder(boolean bRandomOrder) {
		m_bRandomOrder = bRandomOrder;
	} // SetRandomOrder

	/**
	 * Get random order flag
	 * 
	 * @return the random order flag
	 */
	public boolean getRandomOrder() {
		return m_bRandomOrder;
	} // getRandomOrder

}

/**
 * Randomly determines the network structure/graph of the network restricted by
 * its initial structure (which can be an empty graph, or a Naive Bayes graph).
 */
class RandomSearch extends SearchAlgorithm {

	/** for serialization */
	static final long serialVersionUID = 86593475210598631L;

	/** Holds upper bound on number of parents */
	// protected int m_nMaxNrOfParents = 1;
	/**
	 * whether initial structure is an empty graph or a Naive Bayes network
	 */
	// protected boolean m_bInitAsNaiveBayes = true;
	/** Holds flag to indicate ordering should be random * */
	private boolean m_bRandomOrder = true;

	/** Seed for randomization */
	private int m_Seed = (int) (Math.random() * 10000);

	/**
	 * Randomly determines the network structure/graph of the network restricted
	 * by its initial structure (which can be an empty graph, or a Naive Bayes
	 * graph.
	 * 
	 * @param bayesNet
	 *            the network
	 * @param instances
	 *            the data to work with
	 * @throws Exception
	 *             if something goes wrong
	 */
	@Override
	public void search(BayesNet bayesNet, Instances instances) throws Exception {
		int numAttribute = instances.numAttributes();
		int nOrder[] = new int[numAttribute];
		nOrder[0] = instances.classIndex();

		int nAttribute = 0;
		for (int iOrder = 1; iOrder < instances.numAttributes(); iOrder++) {
			if (nAttribute == instances.classIndex()) {
				nAttribute++;
			}
			nOrder[iOrder] = nAttribute++;
		}

		Random random = new Random(m_Seed);
		if (m_bRandomOrder) {
			// generate random ordering (if required)
			int iClass;
			if (getInitAsNaiveBayes()) {
				iClass = 0;
			} else {
				iClass = -1;
			}
			for (int iOrder = 0; iOrder < instances.numAttributes(); iOrder++) {
				int iOrder2 = random.nextInt(numAttribute);
				if (iOrder != iClass && iOrder2 != iClass) {
					int nTmp = nOrder[iOrder];
					nOrder[iOrder] = nOrder[iOrder2];
					nOrder[iOrder2] = nTmp;
				}
			}
		}

		// randomly search restricted by ordering
		for (int iOrder = 1; iOrder < instances.numAttributes(); iOrder++) {
			Set<Integer> parentsAttributes = new HashSet<Integer>(
					getMaxNrOfParents());
			// for (int i = 0; i < getMaxNrOfParents(); i++)
			int numOfParents = (iOrder <= getMaxNrOfParents() ? iOrder
					: getMaxNrOfParents());
			while (parentsAttributes.size() != numOfParents) {
				int randInt = random.nextInt(iOrder);
				if (iOrder > getMaxNrOfParents() && randInt == 0
						&& m_bInitAsNaiveBayes)
					continue;
				parentsAttributes.add(randInt);
			}

			int iAttribute = nOrder[iOrder];
			for (int pIndex : parentsAttributes) {
				int pAttribute = nOrder[pIndex];
				if (!bayesNet.getParentSet(iAttribute).contains(pAttribute))
					bayesNet.getParentSet(iAttribute).addParent(pAttribute,
							instances);
			}
		}
	}

	/**
	 * Sets the max number of parents
	 * 
	 * @param nMaxNrOfParents
	 *            the max number of parents
	 */
	public void setMaxNrOfParents(int nMaxNrOfParents) {
		m_nMaxNrOfParents = nMaxNrOfParents;
	}

	/**
	 * Gets the max number of parents.
	 * 
	 * @return the max number of parents
	 */
	public int getMaxNrOfParents() {
		return m_nMaxNrOfParents;
	}

	/**
	 * Sets whether to init as naive bayes
	 * 
	 * @param bInitAsNaiveBayes
	 *            whether to init as naive bayes
	 */
	public void setInitAsNaiveBayes(boolean bInitAsNaiveBayes) {
		m_bInitAsNaiveBayes = bInitAsNaiveBayes;
	}

	/**
	 * Gets whether to init as naive bayes
	 * 
	 * @return whether to init as naive bayes
	 */
	public boolean getInitAsNaiveBayes() {
		return m_bInitAsNaiveBayes;
	}

	/**
	 * Set random order flag
	 * 
	 * @param bRandomOrder
	 *            the random order flag
	 */
	public void setRandomOrder(boolean bRandomOrder) {
		m_bRandomOrder = bRandomOrder;
	} // SetRandomOrder

	/**
	 * Get random order flag
	 * 
	 * @return the random order flag
	 */
	public boolean getRandomOrder() {
		return m_bRandomOrder;
	} // getRandomOrder

	/**
	 * Set random seed
	 * 
	 */
	public void setSeed(int seed) {
		m_Seed = seed;
	} // setSeed

	/**
	 * Get random seed
	 * 
	 * @return the random seed
	 */
	public int getSeed() {
		return m_Seed;
	} // getSeed
}
