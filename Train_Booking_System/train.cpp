#include <simplecpp>

const int MAX_PASSENGERS = 100;
const int MAX_NAME_LENGTH = 50;
const int SEGMENTS = 10;
const int MAX_AVAILABLE = 10;
const int FARE = 100;

struct Passenger {
    char name[MAX_NAME_LENGTH];
    int fromStation;
    int toStation;
    int groupSize;
};

struct Train {
    int available[SEGMENTS];
    Passenger passengers[MAX_PASSENGERS];
    int passengerCount;

    Train();
    bool bookTickets(const Passenger& p);
    int calculateFare(const Passenger& p);
    int boardingAt(int station, Passenger boardingList[]);
};

Train::Train() {
    for(int i=0; i<10; i++) {
        available[i]=MAX_AVAILABLE;
    }
    passengerCount=0;
}

bool Train::bookTickets(const Passenger& p) {
    for (int i=p.fromStation; i<p.toStation; i++) {
        if (available[i] < p.groupSize) {
            return false; 
        }
    }

    for (int i=p.fromStation; i<p.toStation; i++) {
        available[i] -= p.groupSize;
    }
    passengers[passengerCount] = p;
    passengerCount++;

    return true;
}

int Train::calculateFare(const Passenger& p) {
    int segments = p.toStation - p.fromStation;
    return segments * FARE * p.groupSize;
}

int Train::boardingAt(int station, Passenger boardingList[]) {
    int count = 0;
    for (int i=0; i<passengerCount; i++) {
        if (passengers[i].fromStation==station) {
            boardingList[count]=passengers[i];
            count++;
        }
    }
    return count;
}


main_program {
    int N;
    cin >> N;
    Train train;

    for (int i = 0; i < N; i++) {
        Passenger p;
        
        cin >> p.name >> p.fromStation >> p.toStation >> p.groupSize;

        if (train.bookTickets(p)) {
            cout << train.calculateFare(p) << endl;
        } else {
            cout << "Not available" << endl;
        }
    }

    int stNumber;
    cin >> stNumber;

    Passenger boardingList[MAX_PASSENGERS];
    int boardCount = train.boardingAt(stNumber, boardingList);

    int totalBoarding = 0;
    for (int i = 0; i < boardCount; i++) {
        totalBoarding += boardingList[i].groupSize;
    }

    cout << "Number of passengers boarding at station " << stNumber << ": ";
    cout << totalBoarding << endl;

    for (int i = 0; i < boardCount; i++) {
        cout << boardingList[i].name << " (Group of " << boardingList[i].groupSize << ")" << endl;
    }
}