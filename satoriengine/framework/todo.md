
>> the oldengine.run() needs to be run inorder for the csv to be saved into data :

    * Will this cause a problem ? since the modelmanager is set to empty.

>> baseline requires atleast 3 rows :

    * only run the new engine after 3 rows of data is recieved

                    or

    * make a naive forecaster which outputs the last value ( if only 1 value ) or averages if 2 rows of data

                    or

    * make the oldengine run until there are 3 rows of data inside the aggregate.csv


>> the new engine should only look for aggregate.csv after the message is recieved :

    * keep looking for csv after a certain time 

                    or

    * make some kind of behaviour subject that lets the engine know that the aggregate.csv is produced or updated

>> make the handle prediction function working :

    * handle prediction function inside the start.py is not printing after .predict() is called.
