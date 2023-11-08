#!/usr/bin/python

from collections import namedtuple
import time
import sys

class Edge:
    def __init__ (self, origin=None):
        self.origin = origin
        self.weight = 1
        self.airportListIndex = airportHash[origin]
        
    def __repr__(self):
        return "edge: {0} {1}".format(self.origin, self.weight)
    
    def incWeight(self):
        self.weight += 1
        
    ## write rest of code that you need for this class
    pass

class Airport:
    def __init__ (self, iden=None, name=None):
        self.code = iden
        self.name = name
        self.routes = []
        self.routeHash = dict() # hash key IATA code -> index of edge at self.routes
        self.outweight =  0 # Number of outgoing airport routes

    def __repr__(self):
        return f"{self.code}\t{self.pageIndex}\t{self.name}"
    
    def getEdge(self, originCode):
        return self.routes[self.routeHash[originCode]]
    
    def addIncomingEdge(self, originCode):
        # New incoming edge
        if (not originCode in self.routeHash): 
            self.routeHash[originCode] = len(self.routes)
            self.routes.append(Edge(originCode))
        # Existing incoming edge
        else: 
            self.getEdge(originCode).incWeight()

    def incOutWeight(self):
        self.outweight += 1

#Consts
CONTINUE_PAGE_RANK_THRESHOLD = 0**(-12)
L = 0.9

#Helpers
airportList = [] # list of Airport
airportHash = dict() # hash key IATA code -> index of airport at airportList
finalPageRank = []



def readAirports(fd):
    print("Reading Airport file from {0}".format(fd))
    airportsTxt = open(fd, "r");
    cont = 0
    for line in airportsTxt.readlines():
        a = Airport()
        try:
            temp = line.split(',')
            if len(temp[4]) != 5 :
                raise Exception('not an IATA code')
            a.name=temp[1][1:-1] + ", " + temp[3][1:-1]
            a.code=temp[4][1:-1]
            airportHash[a.code] = cont
        except Exception as inst:
            pass
        else:
            cont += 1
            airportList.append(a)

    airportsTxt.close()
    print(f"There were {cont} Airports with IATA code")


def getAirport(airportCode):
    return airportList[airportHash[airportCode]];

def readRoutes(fd):
    print("Reading Routes file from {fd}")
    routesTxt = open(fd, "r");
    cont = 0;
    for line in routesTxt.readlines():
        try:
            temp = line.split(',');
            if len(temp[2]) != 3 or len(temp[4]) != 3:
                raise Exception('not an IATA code');

            originCode = temp[2]
            destinationCode = temp[4]

            if not originCode in airportHash or not destinationCode in airportHash:
                raise Exception(f"Invalid IATA codes - origin: {originCode} destination: {destinationCode}")
            
            getAirport(destinationCode).addIncomingEdge(originCode)
            getAirport(originCode).incOutweight()

        except Exception as inst:
            print()
        else:
            cont += 1;
    routesTxt.close()
    print(f"There were {cont} routes with IATA code")

def endPageRank(P, Q):
    for x, y in zip(P,Q):
        if (abs(x-y) > CONTINUE_PAGE_RANK_THRESHOLD):
            return False
    return True

def computePageRanks():
    print("Start page rank")
    n = len(airportHash)
    P = [1/n]*n
    end = False
    it = 0
    while (not end):
        Q = [0]*n
        for i in range(n):
            airport = airportList[i]
            pageRank = 0
            for edge in airport.routes:
                pageRank += P[edge.airportListIndex] * edge.weight / airportList[edge.airportListIndex].outweight # P[j] * w(j,i) / out(j)
            Q[i] = L * pageRank + (1-L)/n 
        end = endPageRank(P, Q)
        P = Q
        it += 1
    global finalPageRank
    finalPageRank = P
    print("End page rank")
    return it

def outputPageRanks():
    print("Start output page rank")
    pVector = []
    for i in range(len(airportHash)):
        pVector.append((airportList[i].name, finalPageRank[i]))

    pVector.sort(key = lambda x: x[1], reverse= True)

    for p in pVector:
        print(f"{p[0]} : {p[1]}")

    print("End output page rank")

def main(argv=None):
    readAirports("airports.txt")
    readRoutes("routes.txt")
    time1 = time.time()
    iterations = computePageRanks()
    time2 = time.time()
    outputPageRanks()
    print("#Iterations:", iterations)
    print("Time of computePageRanks():", time2-time1)


if __name__ == "__main__":
    sys.exit(main())
