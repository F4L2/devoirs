#include <iostream>
#include <vector>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <string>
#include <cassert>
using namespace std;


vector<string> split(string stringToBeSplitted, string delimeter){
	std::vector<std::string> splittedString;
	int startIndex = 0;
	int  endIndex = 0;
	while( (endIndex = stringToBeSplitted.find(delimeter, startIndex)) < stringToBeSplitted.size() ){
 
		std::string val = stringToBeSplitted.substr(startIndex, endIndex - startIndex);
		splittedString.push_back(val);
		startIndex = endIndex + delimeter.size();
 
	}
	if(startIndex < stringToBeSplitted.size()){
		std::string val = stringToBeSplitted.substr(startIndex);
		splittedString.push_back(val);
	}
	return splittedString;
}

int main(int argc, char** argv){
    
    if(argc < 2){
        return 1;
    }

    ifstream f;
    f.open (argv[1]);

    ofstream myfile ("output");
 
   
    string line;
    string token;
    int i= 0,j = 0;

    if (f && myfile){
        getline( f, line );

        vector<int> param (3);

        i = 0;
        istringstream ss(line);

        while(getline( ss , token, ' ')) {
            param[i++] = atoi(token.c_str());
        }

        string dic[param[1]];

        for(i = 0; i < param[1]; i++){
            getline( f, line );
            dic[i] = line;
        }

        vector<string> L(param[0]);
        int sum = 0;
        for(i = 0; i < param[2]; i++){
            int l_ind = 0;
            getline( f, line );

            for(j = 0; j< line.size(); j++) {
                if(line[j] == '('){
                    string par = "";
                    j+=1;
                    for(; line[j] != ')'; j++){
                        cout << line[j];
                        par += line[j];
                    }
                    cout << line[j] << '\n';
                    L[l_ind] = par;
                    j += 1;
                }else{
                    L[l_ind] = line[j];
                }
                l_ind ++;
            }

            // calcul sum ... 

            myfile << "Case #" << i+1 << " " << sum << '\n';
        }

        f.close();
        myfile.close();
    }

    

    return 0;
}
