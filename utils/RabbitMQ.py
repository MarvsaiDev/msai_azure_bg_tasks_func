from aio_pika import connect, Message, ExchangeType
from config.rabbit_config import RABBITMQ_SETTINGS

import json


async def publishMsgOnRabbitMQ(msg, email):
    EXCHANGE_NAME = "ai_training_progress"

    connection = await connect(RABBITMQ_SETTINGS)

    channel = await connection.channel()

    # exchange is broker between publisher and queue
    exchange = await channel.declare_exchange(EXCHANGE_NAME, ExchangeType.DIRECT)
    
    # the routing_key is queue name
    await exchange.publish(
        Message(body=json.dumps(msg).encode()),
        routing_key=email
    )

    await connection.close()


async def consumeMsgFromRabbitMQ(email):
    EXCHANGE_NAME = "ai_training_progress"

    connection = await connect(RABBITMQ_SETTINGS)

    channel = await connection.channel()

    exchange = await channel.declare_exchange(EXCHANGE_NAME, ExchangeType.DIRECT)

    queue_name = f'websocket_queue_{email}'
    await channel.queue_delete(queue_name=queue_name)
    
    queue = await channel.declare_queue(queue_name, durable=False)

    await queue.bind(exchange, routing_key=email)


    async for message in queue.iterator():
        async with message.process():
            data = json.dumps(json.loads(message.body.decode()))
            yield data

    