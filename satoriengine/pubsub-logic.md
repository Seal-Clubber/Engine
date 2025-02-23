We have to support pubsub logic as a backup solution in the case of the p2p not working for some setups.

Here's how we'll support pubsub within the code defaulted for p2p:

- Neuron
  - behave mostly the same
  - add an indication for the engine data client to connect to pubsub network for it's data instead
    - I think this looks like passing the entire payload that the neuron uses right now to connect to the pubsub server to the data Server so the engine can pick it up.
  - by the way, the neuron should always publish to pubsub to accomodate those that are subscribing to it.
    - so when the neuron gets data it sends it to the central servers for scoring and pubsub for subscribers.
    - this probably means we have to give it 2 keys, which perhaps we already do: one for just publishing, and one for subscribing.
- Data Manager
  - behaves exactly the same
    - we probably need to add an endpoint for the pubsub payload stuff to push it from neuron and get it to engine
- Engine
  - add an additional call to hit the data server asking for pubsub payload
    - if empty...
      - use p2p
    - if not empty (that we should use pubsub for subscribing) use pubsub...
      - initializes it's data history from it's own data server instead of what it would typically do - try to sync the history from the remote data server.
      - makes connections to pubsub servers
      - sets up subscriptions on the pubsub servers
      - when it gets data (observations) from the pubsub server, it will...
        - conform the data to the correct format (if necessary) and send the data to the data server
        - use the data as normal
