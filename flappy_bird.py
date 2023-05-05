from itertools import cycle
from collections import deque
from pygame.locals import*
from config import config
from agent_training import QLearning
import copy
import random
import sys
import pygame

Agent = QLearning(config['train'])
if Agent.train:
    print("Training model...")
else:
    print("Running model...")

FPS = 30
SCREENWIDTH = 288
SCREENHEIGHT = 512
PIPEGAPSIZE = 100
BASEY = SCREENHEIGHT * 0.79
IMAGES, HITMASKS = {}, {}
STATE_HISTORY = deque(maxlen=70)
REPLAY_BUFFER = []

PLAYERS_LIST = (
    # blue bird
    (
        'assets/sprites/bluebird-upflap.png',
        'assets/sprites/bluebird-midflap.png',
        'assets/sprites/bluebird-downflap.png',
    ),
    # yellow bird
    (
        'assets/sprites/yellowbird-upflap.png',
        'assets/sprites/yellowbird-midflap.png',
        'assets/sprites/yellowbird-downflap.png',
    ),
)

BACKGROUNDS_LIST = (
    'assets/sprites/background.png',
    'assets/sprites/background-black.png',

)

PIPES_LIST = (
    'assets/sprites/pipe-red.png',
)

try:
    xrange
except NameError:
    xrange = range


def main():
    global SCREEN, FPSCLOCK
    pygame.init()
    FPSCLOCK = pygame.time.Clock()
    SCREEN = pygame.display.set_mode((SCREENWIDTH, SCREENHEIGHT))
    pygame.display.set_caption('Flappy Bird')

    IMAGES['numbers'] = (
        pygame.image.load('assets/sprites/0.png').convert_alpha(),
        pygame.image.load('assets/sprites/1.png').convert_alpha(),
        pygame.image.load('assets/sprites/2.png').convert_alpha(),
        pygame.image.load('assets/sprites/3.png').convert_alpha(),
        pygame.image.load('assets/sprites/4.png').convert_alpha(),
        pygame.image.load('assets/sprites/5.png').convert_alpha(),
        pygame.image.load('assets/sprites/6.png').convert_alpha(),
        pygame.image.load('assets/sprites/7.png').convert_alpha(),
        pygame.image.load('assets/sprites/8.png').convert_alpha(),
        pygame.image.load('assets/sprites/9.png').convert_alpha()
    )

    IMAGES['cover'] = pygame.image.load('assets/sprites/cover.png').convert_alpha()
    IMAGES['base'] = pygame.image.load('assets/sprites/base.png').convert_alpha()

    while True:
        randBg = random.randint(0, len(BACKGROUNDS_LIST) - 1)
        IMAGES['background'] = pygame.image.load(BACKGROUNDS_LIST[randBg]).convert()

        randPlayer = random.randint(0, len(PLAYERS_LIST) - 1)
        IMAGES['player'] = (
            pygame.image.load(PLAYERS_LIST[randPlayer][0]).convert_alpha(),
            pygame.image.load(PLAYERS_LIST[randPlayer][1]).convert_alpha(),
            pygame.image.load(PLAYERS_LIST[randPlayer][2]).convert_alpha(),
        )
        pipeindex = random.randint(0, len(PIPES_LIST) - 1)
        IMAGES['pipe'] = (
            pygame.transform.rotate(pygame.image.load(PIPES_LIST[pipeindex]).convert_alpha(), 180),
            pygame.image.load(PIPES_LIST[pipeindex]).convert_alpha(),
        )
        HITMASKS['pipe'] = (
            getHitmask(IMAGES['pipe'][0]),
            getHitmask(IMAGES['pipe'][1]),
        )
        HITMASKS['player'] = (
            getHitmask(IMAGES['player'][0]),
            getHitmask(IMAGES['player'][1]),
            getHitmask(IMAGES['player'][2]),
        )

        movementInfo = showWelcomeAnimation()
        crashInfo = mainGame(movementInfo)
        showGameOverScreen(crashInfo)


def showWelcomeAnimation():
    playery = int((SCREENHEIGHT - IMAGES['player'][0].get_height()) / 2)
    playerIndexGen = cycle([0, 1, 2, 1])
    return {
        'playery': playery,
        'basex': 0,
        'playerIndexGen': playerIndexGen,
    }

def mainGame(movementInfo):
    score = playerIndex = loopIter = 0
    playerIndexGen = movementInfo['playerIndexGen']
    playerx, playery = int(SCREENWIDTH * 0.2), movementInfo['playery']

    basex = movementInfo['basex']
    baseShift = IMAGES['base'].get_width() - IMAGES['background'].get_width()

    newPipe1 = getRandomPipe()
    newPipe2 = getRandomPipe()

    upperPipes = [
        {'x': SCREENWIDTH + 200, 'y': newPipe1[0]['y']},
        {'x': SCREENWIDTH + 200 + (SCREENWIDTH / 2), 'y': newPipe2[0]['y']},
    ]

    lowerPipes = [
        {'x': SCREENWIDTH + 200, 'y': newPipe1[1]['y']},
        {'x': SCREENWIDTH + 200 + (SCREENWIDTH / 2), 'y': newPipe2[1]['y']},
    ]

    pipeVelX = -4
    playerVelY = -9
    playerMaxVelY = 10
    playerMinVelY = -8
    playerAccY = 1
    playerFlapAcc = -9
    playerFlapped = False

    if len(STATE_HISTORY) < 20:
        STATE_HISTORY.clear()
    resume_from_history = len(STATE_HISTORY) > 0 if Agent.train else None
    initial_len_history = len(STATE_HISTORY)
    resume_from = 0
    current_score = STATE_HISTORY[-1][5] if resume_from_history else None
    print_score = False

    while True:
        if resume_from_history:
            if resume_from < initial_len_history:
                if resume_from == 0:
                    playerx, playery, playerVelY, lowerPipes, upperPipes, score, playerIndex = \
                        STATE_HISTORY[resume_from]
                else:
                    lowerPipes, upperPipes = STATE_HISTORY[resume_from][3], STATE_HISTORY[resume_from][4]
                resume_from += 1
        else:
            if Agent.train and config['resume_score'] and score >= config['resume_score']:
                    STATE_HISTORY.append([playerx, playery, playerVelY, copy.deepcopy(lowerPipes),
                                          copy.deepcopy(upperPipes), score, playerIndex])

        for event in pygame.event.get():
            if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                if print_score:
                    print('')
                Agent.save_qvalues()
                Agent.save_training_states()
                pygame.quit()
                sys.exit()
            if event.type == KEYDOWN and (event.key == K_SPACE or event.key == K_UP):
                if playery > -2 * IMAGES['player'][0].get_height():
                    playerVelY = playerFlapAcc
                    playerFlapped = True

        if Agent.act(playerx, playery, playerVelY, lowerPipes):
            if playery > -2 * IMAGES['player'][0].get_height():
                playerVelY = playerFlapAcc
                playerFlapped = True
        crashTest = checkCrash({'x': playerx, 'y': playery, 'index': playerIndex},
                               upperPipes, lowerPipes)
        if crashTest[0]:
            if print_score:
                print('')
            if resume_from_history:
                if score > current_score:
                    Agent.update_qvalues(score)
                else:
                    REPLAY_BUFFER.append(copy.deepcopy(Agent.moves))
                if score > current_score or len(REPLAY_BUFFER) >= 50:
                    random.shuffle(REPLAY_BUFFER)
                    for _ in range(5):
                        if REPLAY_BUFFER:
                            Agent.moves = REPLAY_BUFFER.pop()
                            Agent.update_qvalues(current_score)
                    STATE_HISTORY.clear()
                    REPLAY_BUFFER.clear()
            else:
                Agent.update_qvalues(score)
            if Agent.train:
                print(f"Episode: {Agent.episode}, alpha: {Agent.alpha}, score: {score}, max_score: {Agent.max_score}")
            else:
                print(f"Episode: {Agent.episode}, score: {score}, max_score: {Agent.max_score}")
            return {
                'y': playery,
                'groundCrash': crashTest[1],
                'basex': basex,
                'upperPipes': upperPipes,
                'lowerPipes': lowerPipes,
                'score': score,
                'playerVelY': playerVelY,
            }

        playerMidPos = playerx + IMAGES['player'][0].get_width() / 2
        for pipe in upperPipes:
            pipeMidPos = pipe['x'] + IMAGES['pipe'][0].get_width() / 2
            if pipeMidPos <= playerMidPos < pipeMidPos + 4:
                score += 1
                if score % config['print_score'] == 0:
                    print_score = True
                    print(f"\r {'Training' if Agent.train else 'Running'} agent, "
                          f"score reached (nearest 10,000): {score:,}", end="")
                if config['max_score'] and score >= config['max_score']:
                    if print_score:
                        print('')
                    Agent.end_episode(score)
                    STATE_HISTORY.clear()
                    REPLAY_BUFFER.clear()
                    print(f"Max score of {config['max_score']} reached at episode {Agent.episode}...")
                    return {
                        'y': playery,
                        'groundCrash': crashTest[1],
                        'basex': basex,
                        'upperPipes': upperPipes,
                        'lowerPipes': lowerPipes,
                        'score': score,
                        'playerVelY': playerVelY,
                    }

        if (loopIter + 1) % 3 == 0:
            playerIndex = next(playerIndexGen)
        loopIter = (loopIter + 1) % 30
        basex = -((-basex + 100) % baseShift)

        if playerVelY < playerMaxVelY and not playerFlapped:
            playerVelY += playerAccY
        if playerFlapped:
            playerFlapped = False

        playerHeight = IMAGES['player'][playerIndex].get_height()
        playery += min(playerVelY, BASEY - playery - playerHeight)

        if resume_from >= initial_len_history:
            for uPipe, lPipe in zip(upperPipes, lowerPipes):
                uPipe['x'] += pipeVelX
                lPipe['x'] += pipeVelX

        if 0 < upperPipes[0]['x'] < 5:
            newPipe = getRandomPipe()
            upperPipes.append(newPipe[0])
            lowerPipes.append(newPipe[1])

        if upperPipes[0]['x'] < -IMAGES['pipe'][0].get_width():
            upperPipes.pop(0)
            lowerPipes.pop(0)

        if config['show_game']:
            SCREEN.blit(IMAGES['background'], (0, 0))

            for uPipe, lPipe in zip(upperPipes, lowerPipes):
                SCREEN.blit(IMAGES['pipe'][0], (uPipe['x'], uPipe['y']))
                SCREEN.blit(IMAGES['pipe'][1], (lPipe['x'], lPipe['y']))

            SCREEN.blit(IMAGES['base'], (basex, BASEY))
            showScore(score)

            playerSurface = IMAGES['player'][playerIndex]
            SCREEN.blit(playerSurface, (playerx, playery))

            pygame.display.update()
            FPSCLOCK.tick(FPS)

def showGameOverScreen(crashInfo):
    crashInfo['score']
    playerx = SCREENWIDTH * 0.2
    playery = crashInfo['y']
    playerHeight = IMAGES['player'][0].get_height()
    playerVelY = crashInfo['playerVelY']
    playerAccY = 2
    basex = crashInfo['basex']
    upperPipes, lowerPipes = crashInfo['upperPipes'], crashInfo['lowerPipes']

    while True:
        for event in pygame.event.get():
            if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                Agent.save_qvalues()
                Agent.save_training_states()
                pygame.quit()
                sys.exit()
            if event.type == KEYDOWN and (event.key == K_SPACE or event.key == K_UP):
                if playery + playerHeight >= BASEY - 1:
                    return
        return

def getRandomPipe():
    gapY = random.randrange(0, int(BASEY * 0.6 - PIPEGAPSIZE))
    gapY += int(BASEY * 0.2)
    pipeHeight = IMAGES['pipe'][0].get_height()
    pipeX = SCREENWIDTH + 10

    return [
        {'x': pipeX, 'y': gapY - pipeHeight},
        {'x': pipeX, 'y': gapY + PIPEGAPSIZE},
    ]

def showScore(score):
    scoreDigits = [int(x) for x in list(str(score))]
    totalWidth = 0

    for digit in scoreDigits:
        totalWidth += IMAGES['numbers'][digit].get_width()

    Xoffset = (SCREENWIDTH - totalWidth) / 2

    for digit in scoreDigits:
        SCREEN.blit(IMAGES['numbers'][digit], (Xoffset, SCREENHEIGHT * 0.1))
        Xoffset += IMAGES['numbers'][digit].get_width()

def checkCrash(player, upperPipes, lowerPipes):
    pi = player['index']
    player['w'] = IMAGES['player'][0].get_width()
    player['h'] = IMAGES['player'][0].get_height()

    if player['y'] + player['h'] >= BASEY - 1:
        return [True, True]
    else:

        playerRect = pygame.Rect(player['x'], player['y'],
                                 player['w'], player['h'])
        pipeW = IMAGES['pipe'][0].get_width()
        pipeH = IMAGES['pipe'][0].get_height()

        for uPipe, lPipe in zip(upperPipes, lowerPipes):
            uPipeRect = pygame.Rect(uPipe['x'], uPipe['y'], pipeW, pipeH)
            lPipeRect = pygame.Rect(lPipe['x'], lPipe['y'], pipeW, pipeH)

            pHitMask = HITMASKS['player'][pi]
            uHitmask = HITMASKS['pipe'][0]
            lHitmask = HITMASKS['pipe'][1]

            uCollide = pixelCollision(playerRect, uPipeRect, pHitMask, uHitmask)
            lCollide = pixelCollision(playerRect, lPipeRect, pHitMask, lHitmask)

            if uCollide or lCollide:
                return [True, False]

    return [False, False]

def pixelCollision(rect1, rect2, hitmask1, hitmask2):
    rect = rect1.clip(rect2)

    if rect.width == 0 or rect.height == 0:
        return False

    x1, y1 = rect.x - rect1.x, rect.y - rect1.y
    x2, y2 = rect.x - rect2.x, rect.y - rect2.y

    for x in xrange(rect.width):
        for y in xrange(rect.height):
            if hitmask1[x1 + x][y1 + y] and hitmask2[x2 + x][y2 + y]:
                return True
    return False

def getHitmask(image):
    mask = []
    for x in xrange(image.get_width()):
        mask.append([])
        for y in xrange(image.get_height()):
            mask[x].append(bool(image.get_at((x, y))[3]))
    return mask


if __name__ == '__main__':
    main()
