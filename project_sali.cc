//CLassificatore Naive Bayes.
//Sali Vittorio
#include <string>
#include <fstream>
#include <map>
#include <sstream>
#include <iostream>
using namespace std;

class Document{
	private:
		map<string,int> words;
		string className;
	public:
		Document(const string& name, const string& doc){
			className = name;
			istringstream is(doc);
			string word;
			while(is >> word){//remove dots and commas and other useless characters from the words. Then set the first charcater as lower
				if(word.substr(word.length()-1,1) == "." || word.substr(word.length()-1,1) == "\"" || word.substr(word.length()-1,1) == "," || word.substr(word.length()-1,1) == ";" || word.substr(word.length()-1,1) == ":")
					word.replace(word.length()-1,1,""); 
				if(word.substr(0,1) == "\"")
					word.replace(0,1,"");
				if(isupper(word[0])){
					word[0] += 32;
				}
				words[word]++;
			}
			//riempita in teoria
		}
		inline string getClass() const{ return className; }
		inline map<string,int> getWords() const{ return words; }
		class ConstIterator{//Iterator classes to scan every structure
			private:
				map<string,int>::const_iterator iter;
				map<string,int>::const_iterator finish;
			public:
				ConstIterator(const Document& d){
					iter = d.words.begin();
					finish = d.words.end();
				}
				const pair<string,int> getNext(){
					const pair<string,int> p = *iter;
					iter++;
					return p;
				}
				inline bool hasNext() const{
					if(iter == finish)
						return false;
					return true;
				}
		};
		friend class ConstIterator;
		friend bool operator==(const Document& a, const Document& b);
		friend struct Cmp;
		friend ostream& operator<<(ostream& out, const Document& d);
};

bool operator==(const Document& a, const Document& b){
	if(a.className != b.className)
		return false;
	if(a.words.size() != b.words.size())
		return false;
	if(a.words != b.words)
		return false;
	return true;
}

ostream& operator<<(ostream& out, const Document& d){
	out << d.className << endl;
	for(auto iter = d.words.begin(); iter != d.words.end() ; iter++)
		out << iter->first << "\t" << iter->second << endl;;
	out << endl;
	return out;
}

struct Cmp{
	bool operator()(const Document& a, const Document& b) const{//order by class and then by number of words
		if(a.className != b.className)
			return a.className < b.className;
		return a.words.size() < b.words.size();
	}
};

class TrainingSet{
	private:
		map<Document,string,Cmp> examples;// document and topic (macro class to classify)
	public:
		TrainingSet(const string& filename){ //read from a file
			string topic;
			string d;
			ifstream is(filename);
			if(is.good()){
				while(is >> topic && getline(is,d)){
					examples[Document(topic,d)] = topic;
				}
			} else {
				cerr << "file not open correctly" << endl;
			}
		}
		inline map<Document,string,Cmp> getExamples() const{ return examples; }
		class ConstIterator{//An other iterator
			private:
				map<Document,string>::const_iterator iter;
				map<Document,string>::const_iterator finish;
			public:
				ConstIterator(const TrainingSet& t){
					iter = t.examples.begin();
					finish = t.examples.end();
				}
				const pair<Document,string> getNext(){
					const pair<Document,string> p = *iter;
					iter++;
					return p;
				}
				inline bool hasNext() const{
					if(iter == finish)
						return false;
					return true;
				}
		};
		friend class ConstIterator;
		friend ostream& operator<<(ostream& out, const TrainingSet& t);
};

ostream& operator<<(ostream& out, const TrainingSet& t){
	TrainingSet::ConstIterator iter(t);
	for(; iter.hasNext() ;)
		out << iter.getNext().first;
	return out;
}

template<typename A, typename B>
class ConstIterator{//An other Iterator
	private:
		typename map<A,B>::const_iterator iter;
		typename map<A,B>::const_iterator finish;
	public:
		ConstIterator(const map<A,B>& m){
			iter = m.begin();
			finish = m.end();
		}
		const pair<A,B> getNext(){
			const pair<A,B> p = *iter;
			iter++;
			return p;
		}
		inline bool hasNext() const{
			if(iter == finish)
				return false;
			return true;
		}
};

class Classifier{ //abstract
	public:
		virtual string classify(const pair<Document,string>& p) const= 0;
		virtual ~Classifier(){}
};

class ClassifierNaiveBayes : public Classifier{
	private:
		TrainingSet t;
		int wordIsIn(const string& w, const Document& doc) const{
			Document::ConstIterator id(doc);
			for(; id.hasNext() ; ){
				auto valued = id.getNext();	
				if(w == valued.first)
					return valued.second;
			}
			return 0;
		}
		string maximumProbability(const map<string,double>& prob) const{
			multimap<double,string> ordered;
			auto iter = prob.begin();
			for(; iter != prob.end() ; iter++)
				ordered.insert(make_pair(iter->second,iter->first));
			auto i = ordered.rbegin();
			return i->second;
		}
	public:
		ClassifierNaiveBayes(const string& filename): t(filename){}
		/*Alghoritm by Bayesian statistics*/
		string classify(const pair<Document,string>& doc) const{
			TrainingSet::ConstIterator iter(t);
			map<string,double> classes;
			int totDoc = 0;					
			for( ; iter.hasNext() ; ){
				classes[iter.getNext().second] = 0.0; //It contain every class
				totDoc++;//number of total documents;
			}
			ConstIterator ic1(classes);
			for(const pair<string,double>& p : classes){//p pair class, probability. Class mean topic
				TrainingSet::ConstIterator iter2(t);
				for( ; iter2.hasNext() ; ){
					auto value = iter2.getNext();//value is the element of the set
					if(value.second == p.first)
						classes[p.first]++; // coupled with class name there are numbers of document of that class 
				}
			}
			ConstIterator ic2(classes);
			for(; ic2.hasNext() ;){
				auto value = ic2.getNext();
				value.second = value.second / totDoc;//prior probability
			}
			
				
			map<string,map<string,double>> wordCount;
			
			Document::ConstIterator itw(doc.first);
			for(; itw.hasNext() ; ){//cycle for classes
				auto value = itw.getNext();
				ConstIterator ic3(classes);
				for(; ic3.hasNext() ; ){//cycle for words
					double count = 0;
					auto valuec = ic3.getNext();
					TrainingSet::ConstIterator iter3(t);
					for(; iter3.hasNext() ; ){//cycle for documents
						auto valued = iter3.getNext();
						if(valued.second == valuec.first)
							count += wordIsIn(value.first,valued.first);
					}
					wordCount[value.first][valuec.first] = count;//times where w come up
				}
				
			}
			map<string,int> classTotalOccurrence;
			ConstIterator ic4(classes);
			int sum;
			for(; ic4.hasNext() ; ){//cycle for classes
				sum = 0;
				auto valuec = ic4.getNext();
				TrainingSet::ConstIterator iter4(t);
				for(; iter4.hasNext() ; ){//cycle for every doc of the set
					auto value = iter4.getNext();
					if(valuec.first == value.second){//if same class
						Document::ConstIterator idw(value.first);
						for(; idw.hasNext() ; ){//cycle for every word of the document
							auto w = idw.getNext();
							sum += w.second;
						}
					 }
				}
				classTotalOccurrence[valuec.first] = sum;//total number of occurrence for each class
			}
			map<string,double> probClassCondit;
			ConstIterator ic5(classes);
			for(; ic5.hasNext() ; ){//cycle for classes
				auto valuec = ic5.getNext();
				probClassCondit[valuec.first] = 1.0;
				Document::ConstIterator itw2(doc.first);
				for(; itw2.hasNext(); ){//cycle for words
					auto w = itw2.getNext();
					if(wordCount[w.first][valuec.first] > 0)
						probClassCondit[valuec.first] *= wordCount[w.first][valuec.first]/classTotalOccurrence[valuec.first];
					else
						probClassCondit[valuec.first] *= 0.00001/classTotalOccurrence[valuec.first];
						//this number 0.00001 is only a very little number that has to be different from zero beacause if a word as no
						//occurrence (wordCount), in this formula, the value of probClassCondit will be zero and the product cancel all the other
						//probabilities. 
				}
			}
			
			/*p(c|d) viene calcolata per ogni classe. Alla fine il documento viene assegnato alla classe
			tale che p(c|d) e' 	massimo.
			*/
			//in the and document was assigned to the class that has  maximum p(c|d) (p = probability, c = class, d = document)

			auto iterProb = probClassCondit.begin();
			auto iterClasses = classes.begin();
			for(; iterProb != probClassCondit.end() ; iterProb++){
				iterProb->second = iterProb->second * iterClasses->second;
				cout << iterProb->first << "\t" << iterProb->second << endl;
				iterClasses++; 
			}
			return maximumProbability(probClassCondit);
		}
		virtual ~ClassifierNaiveBayes(){}
};
int main(){
	ClassifierNaiveBayes c("trainingSet.txt"); /*training set is in italian and also the test, but you can modify your dataset 
	adding text in your favourite language*/
	//format    "Topic(class)", "text correlated to this topic-class"
	Document d1("AUTO", "Il nuovo motore diesel di Fiat è molto potente, anche le nuovissime sospensioni da vendere anche alle altre marche, sono molto interessanti");
	Document d2("SCIENZA", "Nello spazio è possibile trovare un esemplare di buco nero, il quale rompe il tessuto spazio temporale e crea gravità attorno a sè facendo orbitare i corpi vicini attorno a sè stesso");
	Document d3("CALCIO", "Il Barcellona sta conducendo delle trattative di mercato su Vlahovic della Juventus, il loro obbiettivo è vincere la Champions");
	Document d4("CUCINA", "la mia salsa preferita è il guacamole, non necessita di alcuna cottura e gli ingredienti sono veramente semplici da trovare, bastano pomodoro, avocado, olio extravergine d'oliva e peperoncino");
	cout << c.classify(make_pair(d1,d1.getClass())) << endl;
	cout << c.classify(make_pair(d2,d2.getClass())) << endl;
	cout << c.classify(make_pair(d3,d3.getClass())) << endl;
	cout << c.classify(make_pair(d4,d4.getClass())) << endl;
}
