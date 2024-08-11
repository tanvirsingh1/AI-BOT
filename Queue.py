# Copy over your a1_partc.py file here
class Stack:
    def __init__(self, cap=10):
        """
        Constructor for the Stack class.

        parameter cap: Initial cap of the stack. Default cap is 10.
        """
        self.data = [None] * cap
        self.cap = cap
        self.used = 0

    def capacity(self):
        """
        Returns the cap of the stack.

        """
        return self.cap

    def push(self, data):
        """
        Pushes data to the top of the stack. If the stack is full, it increases the cap.

        """
        if self.used == self.cap:
            self.grow()
        self.data[self.used] = data
        self.used += 1

    def pop(self):
        """
        Removes and returns the top element from the stack.

        Raises IndexError: If the stack is empty.

        """
        if self.is_empty():
            raise IndexError('pop() used on empty stack')
        self.used -= 1
        return self.data[self.used]

    def get_top(self):
        """
        Returns the element at the top of the stack.

        """
        return self.data[self.used - 1]

    def is_empty(self):
        """
        Checks if the stack is empty.

        """
        return self.used == 0

    def __len__(self):
        """
        Returns the number of values in the stack.

        """
        return self.used

    def grow(self):
        """
        Increases the cap of the stack by creating a new list with twice the cap.

        """
        temp = [None] * (self.cap * 2)

        for i in range(self.used):
            temp[i] = self.data[i]

        self.data = temp
        self.cap = self.cap * 2


# Queue is FIFO
# Queue Class Implementation
# Author: Tanvir Singh

class Queue:
    def __init__(self, cap=10):
        """Initialize the queue with a specified capacity.

        Args:
            cap: The capacity of the queue.
        """
        self.theQueue = [None] * cap
        self.cap = cap
        self.used = 0
        self.front = 0
        self.back = 0

    def capacity(self):
        """Return the current capacity of the queue."""
        return self.cap

    def enqueue(self, data):
        """Add an element to the back of the queue.

        Args:
            data: The data to be added to the queue.
        """
        if self.used == self.cap:
            self.grow()

        self.theQueue[self.back] = data
        self.back = (self.back + 1) % self.cap
        self.used += 1

    def dequeue(self):
        """Remove and return the element from the front of the queue.

        Raises:
            IndexError: If the queue is empty.

        Returns:
            The data that was at the front of the queue.
        """
        if not self.is_empty():
            d_data = self.theQueue[self.front]
            self.theQueue[self.front] = None
            self.front = (self.front + 1) % self.cap
            self.used -= 1
            return d_data
        else:
            raise IndexError('dequeue() used on empty queue')

    def get_front(self):

        return self.theQueue[self.front]

    def is_empty(self):

        return self.used == 0

    def __len__(self):

        return self.used

    def grow(self):
        """Resize the queue when it reaches its capacity."""
        First = self.front

        # Create a new queue with double the current capacity
        resized_Queue = [None] * (self.cap * 2)

        # Copy elements to the new queue
        for i in range(self.used):
            resized_Queue[i] = self.theQueue[First]
            First = (First + 1) % self.cap

        # Reset front and back pointers
        self.front = 0
        self.back = self.used

        # Update capacity and queue reference
        self.cap *= 2
        self.theQueue = resized_Queue


# Author: Meetsimar Kaur

class Deque:

    def __init__(self, cap=10):
        """
        Initializes a Deque with the specified cap.

        """
        self.the_deque = [None] * cap
        self.cap = cap
        self.used = 0
        self.front = 0  # Index of the oldest item in the deque
        self.back = self.cap - 1

    def capacity(self):
        """
        Returns the current cap of the Deque.
        """
        return self.cap

    def push_front(self, data):
        """
        Adds an item to the front of the Deque.

        If the Deque is full, it increases its cap by doubling.

        """
        if self.used == self.cap:
            self.grow()
        self.the_deque[(self.front - 1) % self.cap] = data
        self.front = (self.front - 1) % self.cap
        self.used += 1

    def push_back(self, data):
        """
        Adds an item to the back of the Deque.

        If the Deque is full, it increases its cap by doubling.

        """
        if self.used == self.cap:
            self.grow()
        self.the_deque[(self.back + 1) % self.cap] = data
        self.back = (self.back + 1) % self.cap
        self.used += 1

    def pop_front(self):
        """
        Removes and returns the item from the front of the Deque.

        Raises IndexError: If the Deque is empty.
        """
        if self.is_empty():
            raise IndexError('pop_front() used on empty deque')

        data = self.the_deque[self.front]
        self.the_deque[self.front] = None
        self.front = (self.front + 1) % self.cap
        self.used -= 1
        return data

    def pop_back(self):
        """
        Removes and returns the item from the back of the Deque.

        Raises IndexError: If the Deque is empty.
        """
        if self.is_empty():
            raise IndexError('pop_back() used on empty deque')

        data = self.the_deque[self.back]
        self.the_deque[self.back] = None
        self.back = (self.back - 1) % self.cap
        self.used -= 1
        return data

    def get_front(self):
        """
        Returns the item at the front of the Deque.

        """
        return self.the_deque[self.front]

    def get_back(self):
        """
        Returns the item at the back of the Deque.

        """
        return self.the_deque[self.back]

    def is_empty(self):
        """
        Checks if the Deque is empty.

        """
        return self.used == 0

    def __len__(self):
        """
        Returns the number of items in the Deque.

        """
        return self.used

    def __getitem__(self, k):
        """
        Returns the item at the specified index in the Deque.
        Raises IndexError: If the index is out of range.
        """
        if k >= self.used:
            raise IndexError('Index out of range')
        return self.the_deque[(self.front + k) % self.cap]

    def grow(self):
        """
        Increases the cap of the Deque by doubling its current cap.
        """
        temp = [None] * (self.cap * 2)

        for i in range(self.used):
            temp[i] = self.the_deque[self.front]
            self.front = (self.front + 1) % self.cap

        self.front = 0
        self.back = self.used - 1
        self.cap *= 2
        self.the_deque = temp