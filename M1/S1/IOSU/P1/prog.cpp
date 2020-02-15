#include <iostream>
#include <vector>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <algorithm>
using namespace std;


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

        int T = atoi(line.c_str());
        vector<int> output(T);
        vector<int>NM (2);
            
        i = 0;
        getline( f, line );
        istringstream ss(line);
        while(getline( ss , token, ' ')) {
            NM[i++] = atoi(token.c_str());
        }

        while (getline( f, line )){
            vector<string> N(NM[0]);

            vector<string> last_dir(NM[0]);

            for(i = 0; i < NM[0]; i++){
                getline( f, line );
                N[i] = line;

                istringstream dir(line);
                string tmp = "";
                while(getline( dir , token, '/')) {
                    tmp = "/" + token;
                }
                last_dir.push_back(tmp);
            }   

            int sum = 0;
            for(i = 0; i < NM[1]; i++){
                getline( f, line );
                if (find(N.begin(), N.end(), line) != N.end()){
                    continue;
                }
                istringstream new_dir(line);
                string test_dir = "";
                
                while(getline( new_dir , token, '/')) {
                    if(token.size() == 0){
                        continue;
                    }else{
                        test_dir += "/" + token;
                    }
                    if (find(N.begin(), N.end(), test_dir) != N.end() || find(last_dir.begin(), last_dir.end(), test_dir) != last_dir.end()){
                        continue;
                    }
                    sum += 1;
                }
            }

            myfile << "Case #" << j+1 << " "<< sum << '\n';
            output[j] = sum;
            j++;

            N.clear();
            last_dir.clear();
            
            i = 0;
            istringstream ss(line);
            while(getline( ss , token, ' ')) {
                NM[i++] = atoi(token.c_str());
            }  
        }
        
        for(j = 0; j< output.size(); j++){
            cout << output[j] << ' ';
        }
        
        f.close();
        myfile.close();
    }

    return 0;
}
