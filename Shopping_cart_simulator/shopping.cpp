#include <simplecpp>

const int MAX_ITEMS = 100;

struct Firecracker
{
    string name;
    int price;
    int quantity;
    Firecracker(string n = "", int p = 0, int q = 0) : name(n), price(p), quantity(q) {}
};

class ShoppingCart
{
    Firecracker cart[MAX_ITEMS];
    int quantities[MAX_ITEMS];
    int size;

public:
    ShoppingCart(){
        size = 0;
        for (int i = 0; i < MAX_ITEMS; i++)
            quantities[i] = 0;
    }

    bool add_firecracker(Firecracker &firecracker, int quantity);
    bool delete_firecracker(const string &name);
    void view_cart(){
        if (size == 0)
        {
            cout << "Empty\n";
            return;
        }
        int total = 0;
        for (int i = 0; i < size; i++)
        {
            int cost = quantities[i] * cart[i].price;
            cout << cart[i].name << ": " << quantities[i] << " * " << cart[i].price << " = " << cost << "\n";
            total += cost;
        }
    }
    int checkout();
};
bool ShoppingCart::add_firecracker(Firecracker &firecracker,int quantity){
    if(quantity<= 0 || quantity >firecracker.quantity) 
        return false;

    for (int i = 0; i < size; i++){
        if (cart[i].name==firecracker.name){
            if (quantities[i]+quantity >firecracker.quantity)
                return false;
            quantities[i] += quantity;
            return true;
        }
    }
    if (size<MAX_ITEMS){
        cart[size]=firecracker;
        quantities[size]=quantity;
        size++;
        return true;
    }
    return false;

int ShoppingCart::checkout(){
    if(size==0){
        cout <<"Empty\n";
        return 0;
    }
    int total=0;
    for(int i=0; i<size;i++){
        total+=quantities[i]*cart[i].price;
    }
    cout<<"Total:"<<total<<"\n";
    size=0; 
    return total;
}

main_program
{
    Firecracker stock[] = {
        Firecracker("Anar", 50, 100),
        Firecracker("Phuljhari", 20, 200),
        Firecracker("Chakri", 30, 150),
        Firecracker("Rocket", 100, 50),
        Firecracker("Bomb", 75, 80)};

    const int STOCK_SIZE = sizeof(stock) / sizeof(stock[0]);

    ShoppingCart cart;

    string command;
    while (true){
        getline(cin, command);
        if (command.empty())
            continue;

        if (command == "exit"){
            cout << "Exiting the program.";
            break;
        }

        if (command == "checkout"){
            cart.checkout();
            continue;
        }

        size_t pos = command.find(' ');
        string action = command.substr(0, pos);

        if (action == "add"){
            size_t pos2 = command.find(' ', pos + 1);
            string name = command.substr(pos + 1, pos2 - pos - 1);
            int quantity = stoi(command.substr(pos2 + 1));

            for (int i = 0; i < STOCK_SIZE; i++)
            {
                if (stock[i].name == name)
                {
                    bool added = cart.add_firecracker(stock[i], quantity);
                    if (added)
                    {
                        cout << quantity << " " << name << " added to cart.\n";
                        cart.view_cart();
                    }
                    else
                    {
                        cout << "-1\n";
                        cout << "Empty\n";
                    }
                    break;
                }
            }
        }
        else if (action == "delete"){
            string name = command.substr(pos + 1);
            bool deleted = cart.delete_firecracker(name);
            if (deleted)
            {
                cout << name << " deleted\n";
                cart.view_cart();
            }
            else
            {
                cout << "-1\nEmpty\n";
            }
        }
    }
}